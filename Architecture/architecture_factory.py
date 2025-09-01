import importlib  
from typing import Dict, Any  
from tensorflow.keras.layers import Layer 

def get_architecture(architecture_name: str, **kwargs) -> Layer:  
    """Factory function to create architecture instances"""  
      
    architecture_map = {  
        'neural_receiver1': 'Architecture.neural_receiver1.NeuralReceiver',  
        'neural_receiver2': 'Architecture.neural_receiver2.NeuralReceiverInception',  
        'neural_receiver3': 'Architecture.neural_receiver3.NeuralReceiverSE',  
        'neural_receiver4': 'Architecture.neural_receiver4.NeuralReceiverCNNLSTM',
    }  
      
    if architecture_name not in architecture_map:  
        raise ValueError(f"Unknown architecture: {architecture_name}. Available: {list(architecture_map.keys())}")  
      
    # Import the architecture class dynamically  
    module_path, class_name = architecture_map[architecture_name].rsplit('.', 1)  
    module = importlib.import_module(module_path)  
    architecture_class = getattr(module, class_name)  
      
    # Create instance with provided parameters  
    return architecture_class(**kwargs)