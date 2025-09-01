import yaml  
import os  
from dataclasses import dataclass  
from typing import List, Dict, Any  
  
@dataclass  
class ArchitectureConfig:  
    name: str  

  
@dataclass  
class TrainingConfig:  
    num_training_iterations: int  
    batch_size: int  
    learning_rate: float  
    optimizer: str  
    resume_training: bool = False
  
@dataclass  
class ChannelConfig:  
    carrier_frequency: float  
    delay_spread: float  
    cdl_model: str  
    speed: float  
  
@dataclass  
class SNRConfig:  
    ebno_db_min: float  
    ebno_db_max: float  
  
@dataclass  
class OFDMConfig:  
    subcarrier_spacing: float  
    fft_size: int  
    num_ofdm_symbols: int  
    dc_null: bool  
    num_guard_carriers: List[int]  
    pilot_pattern: str  
    pilot_ofdm_symbol_indices: List[int]  
    cyclic_prefix_length: int  
  
@dataclass  
class ModulationConfig:  
    num_bits_per_symbol: int  
    coderate: float  
  
@dataclass  
class PathsConfig:  
    weights_dir: str  
    results_dir: str  
  
@dataclass  
class EvaluationConfig:  
    ebno_range: List[float]  
    batch_size: int  
    num_target_block_errors: int  
    max_mc_iter: int  
  
@dataclass  
class ExperimentConfig:  
    architecture: ArchitectureConfig  
    training: TrainingConfig  
    channel: ChannelConfig  
    snr: SNRConfig  
    ofdm: OFDMConfig  
    modulation: ModulationConfig  
    paths: PathsConfig  
    evaluation: EvaluationConfig  
  
def load_config(config_path: str = "Config/config.yaml") -> ExperimentConfig:  
    """Load configuration from YAML file"""  
      
    if not os.path.exists(config_path):  
        raise FileNotFoundError(f"Configuration file not found: {config_path}")  
      
    with open(config_path, 'r') as f:  
        config_dict = yaml.safe_load(f)  
      
    # Create configuration objects  
    architecture = ArchitectureConfig(**config_dict['architecture'])  
    training = TrainingConfig(**config_dict['training'])  
    channel = ChannelConfig(**config_dict['channel'])  
    snr = SNRConfig(**config_dict['snr'])  
    ofdm = OFDMConfig(**config_dict['ofdm'])  
    modulation = ModulationConfig(**config_dict['modulation'])  
    paths = PathsConfig(**config_dict['paths'])  
    evaluation = EvaluationConfig(**config_dict['evaluation'])  
      
    return ExperimentConfig(  
        architecture=architecture,  
        training=training,  
        channel=channel,  
        snr=snr,  
        ofdm=ofdm,  
        modulation=modulation,  
        paths=paths,  
        evaluation=evaluation  
    )  
  
def save_config(config: ExperimentConfig, config_path: str = "Config/config.yaml"):  
    """Save configuration to YAML file"""  
      
    config_dict = {  
        'architecture': config.architecture.__dict__,  
        'training': config.training.__dict__,  
        'channel': config.channel.__dict__,  
        'snr': config.snr.__dict__,  
        'ofdm': config.ofdm.__dict__,  
        'modulation': config.modulation.__dict__,  
        'paths': config.paths.__dict__,  
        'evaluation': config.evaluation.__dict__  
    }  
      
    os.makedirs(os.path.dirname(config_path), exist_ok=True)  
    with open(config_path, 'w') as f:  
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
def get_modulation_name(num_bits_per_symbol: int) -> str:  
    """Map number of bits per symbol to modulation name"""  
    modulation_map = {  
        1: "bpsk",  
        2: "qpsk",   
        4: "qam16",  
        6: "qam64",  
        8: "qam256"  
    }  
      
    if num_bits_per_symbol not in modulation_map:  
        raise ValueError(f"Unsupported modulation: {num_bits_per_symbol} bits per symbol")  
      
    return modulation_map[num_bits_per_symbol]