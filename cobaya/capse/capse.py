from typing import Sequence, List

import os
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import json
import importlib.util

from cobaya.log import LoggedError
from cobaya.theory import Theory
from cobaya.theories.cosmo import BoltzmannBase

jax.config.update("jax_enable_x64", True)

def get_flax_params(nn_dict, weights):
  """
  Get the parameters for each layer in a neural network.

  Args:
    nn_dict (dict): A dictionary containing information about the neural network.
    weights (list): A list of weights.

  Returns:
    dict: A dictionary where the keys are the layer names and the values are the parameters.

  """
  in_array, out_array = get_in_out_arrays(nn_dict)
  i_array = get_i_array(in_array, out_array)
  params = [get_weight_bias(i_array[j], in_array[j], out_array[j], weights, nn_dict) for j in range(nn_dict["n_hidden_layers"]+1)]
  layer = ["layer_" + str(j) for j in range(nn_dict["n_hidden_layers"]+1)]
  return dict(zip(layer, params))

def get_weight_bias(i, n_in, n_out, weight_bias):
  """
  Extracts the weight and bias from the given weight_bias array.

  Args:
    i (int): Starting index in the weight_bias array.
    n_in (int): Number of input units.
    n_out (int): Number of output units.
    weight_bias (numpy.ndarray): Array containing weights and biases.

  Returns:
    dict: A dictionary containing the extracted weight and bias.
    int: The updated index after extracting the weight and bias.
  """
  weight = np.reshape(weight_bias[i:i+n_out*n_in], (n_in, n_out))
  bias = weight_bias[i+n_out*n_in:i+n_out*n_in+n_out]
  i += n_out*n_in+n_out
  return {'kernel': weight, 'bias': bias}, i

def get_in_out_arrays(nn_dict):
  """
  Get the input and output arrays for each layer in a neural network.

  Parameters:
  nn_dict (dict): A dictionary containing the neural network configuration.

  Returns:
  tuple: A tuple containing the input array and output array for each layer.

  """
  n = nn_dict["n_hidden_layers"]
  in_array = np.zeros(n+1, dtype=int)
  out_array = np.zeros(n+1, dtype=int)
  in_array[0] = nn_dict["n_input_features"]
  out_array[-1] = nn_dict["n_output_features"]
  for i in range(n):
    in_array[i+1] = nn_dict["layers"]["layer_" + str(i+1)]["n_neurons"]
    out_array[i] = nn_dict["layers"]["layer_" + str(i+1)]["n_neurons"]
  return in_array, out_array

def get_i_array(in_array, out_array):
    i_array = np.empty_like(in_array)
    i_array[0] = 0
    for i in range(1, len(i_array)):
        i_array[i] = i_array[i-1] + in_array[i-1]*out_array[i-1] + out_array[i-1]
    return i_array

def load_weights(nn_dict, weights):
    in_array, out_array = get_in_out_arrays(nn_dict)
    i_array = get_i_array(in_array, out_array)
    variables = {'params': {}}
    i = 0
    for j in range(nn_dict["n_hidden_layers"]+1):
        layer_params, i = get_weight_bias(i_array[j], in_array[j], out_array[j], weights, nn_dict)
        variables['params']["Dense_" + str(j)] = layer_params
    return variables


class MLP(nn.Module):
  """
  Multi-Layer Perceptron (MLP) class.

  Args:
    features (Sequence[int]): List of integers representing the number of units in each layer.
    activations (List[str]): List of activation functions for each layer.
    in_MinMax (np.array): Array representing the minimum and maximum values for input normalization.
    out_MinMax (np.array): Array representing the minimum and maximum values for output denormalization.
    NN_params (dict): Dictionary containing the parameters of the neural network.
    postprocessing (callable, optional): Callable function for postprocessing the model output.

  Methods:
    __call__(self, x): Forward pass of the MLP.
    maximin_input(self, input): Normalize the input using the minimum and maximum values.
    inv_maximin_output(self, output): Denormalize the output using the minimum and maximum values.
    get_Cl(self, input): Get the processed model output.

  """

  features: Sequence[int]
  activations: List[str]
  in_MinMax: np.array
  out_MinMax: np.array
  NN_params: dict
  postprocessing: callable = None

  @nn.compact
  def __call__(self, x):
    """
    Forward pass of the MLP.

    Args:
      x (ndarray): Input data.

    Returns:
      ndarray: Output of the MLP.

    """
    for i, feat in enumerate(self.features[:-1]):
      if self.activations[i] == "tanh":
        x = nn.tanh(nn.Dense(feat)(x))
      elif self.activations[i] == "relu":
        x = nn.relu(nn.Dense(feat)(x))
      # Add more activation functions as needed
    x = nn.Dense(self.features[-1])(x)
    return x

  def maximin_input(self, input):
    """
    Normalize the input using the minimum and maximum values.

    Args:
      input (ndarray): Input data.

    Returns:
      ndarray: Normalized input.

    """
    return (input - self.in_MinMax[:,0]) / (self.in_MinMax[:,1] - self.in_MinMax[:,0])

  def inv_maximin_output(self, output):
    """
    Denormalize the output using the minimum and maximum values.

    Args:
      output (ndarray): Output data.

    Returns:
      ndarray: Denormalized output.

    """
    return output * (self.out_MinMax[:,1] - self.out_MinMax[:,0]) + self.out_MinMax[:,0]
        
  def get_Cl(self, input):
    """
    Get the processed model output.

    Args:
      input (ndarray): Input data.

    Returns:
      ndarray: Processed model output.

    """
    norm_input = self.maximin_input(input)
    norm_model_output = self.apply(self.NN_params, norm_input)
    model_output = self.inv_maximin_output(norm_model_output)
    processed_model_output = self.postprocessing(input, model_output)
    #here we are also postprocessing the Cls, according to what was done in Capse.jl release paper
    return processed_model_output
  

class Capse(Theory):
  """
  Wraps a Flax-based Capse emulator for the CMB power spectra into a Cobaya theory class.
  """

  def capse_pars_to_cobaya_pars(self):
    return {
            "ln10As": "logA",
            "ns": "ns",
            "H0": "H0",
            "ωb": "ombh2",
            "ωc": "omch2",
            "τ": "tau",
        }


  def initialize(self) -> None:
    super().initialize()  

    # Throw a LoggedError if the emulator path is not provided
    if not self.emulator_path:
      raise LoggedError(self.log, "Emulator path not provided.")
    else:
       self.log.info("Loading Capse emulator from %s", self.emulator_path)

    # Loading weights and parameters
    self.in_MinMax = jnp.load(os.path.join(self.emulator_path, self.emulator_settings["inMinMax"]))
    f = open(os.path.join(self.emulator_path, self.emulator_settings["NN_params"]))
    self.NN_dict = json.load(f) # returns JSON object as a dictionary
    f.close()

    spec = importlib.util.spec_from_file_location("test", os.path.join(self.emulator_path, "test.py"))
    self.test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(self.test)

    self.weights = {}
    self.out_MinMax = {}
    self.raw_cls = {}
    self.variables = {}
    self.emulator = {}


    for spectrum in self.emulator_settings["spectra"]:
      self.weights[spectrum] = jnp.load(os.path.join(self.emulator_path, self.emulator_settings[spectrum]["weights"]))
      self.out_MinMax[spectrum] = jnp.load(os.path.join(self.emulator_path, self.emulator_settings[spectrum]["outMinMax"]))
      self.raw_cls[spectrum] = jnp.load(os.path.join(self.emulator_path, self.emulator_settings[spectrum]["raw_cls"]))
      self.variables[spectrum] = load_weights(self.NN_dict, self.weights[spectrum])

      # Instantiating the MLP with the correct architecture
      self.emulator[spectrum] = MLP([self.NN_dict['layers'][l]['n_neurons']  for l in self.NN_dict['layers']] + [self.NN_dict['n_output_features']],
                                    [self.NN_dict['layers'][l]['activation_function']  for l in self.NN_dict['layers']],
                                    self.in_MinMax, 
                                    self.out_MinMax[spectrum], 
                                    self.variables[spectrum], 
                                    self.test.postprocessing)


  def calculate(self, state: dict, want_derived: bool=False, **params) -> bool:
    pass


  def get_Cl(self, ell_factor: bool=False, units: str="FIRASmuK2") -> dict:
    """
      Returns a dictionary of lensed CMB power spectra and the lensing potential ``pp``
      power spectrum from the emulator.
    """
    cls_tmp = self.current_state.copy()