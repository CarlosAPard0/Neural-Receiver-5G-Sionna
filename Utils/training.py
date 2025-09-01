
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"     # Seleccionar GPU 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'     # Silenciar logs verbosos

# (Luego sí) importa TensorFlow una sola vez
import tensorflow as tf

# Verificación temprana y configuración de memoria
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')  # usar solo la 0
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("[OK] GPU visible para TensorFlow:", gpus[0])
    except Exception as e:
        print("[WARN] No se pudo configurar memory_growth:", e)
else:
    print("[ERR] No hay GPUs visibles para TensorFlow")
import tensorflow as tf  
import pickle  
import numpy as np  
from datetime import datetime  


from sionna.phy import Block  
from sionna.phy.channel.tr38901 import Antenna, AntennaArray, CDL  
from sionna.phy.channel import OFDMChannel  
from sionna.phy.mimo import StreamManagement  
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, ResourceGridDemapper  
from sionna.phy.utils import ebnodb2no, insert_dims, expand_to_rank  
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, BinarySource  
  
from Utils.config_loader import load_config, ExperimentConfig, get_modulation_name

from Architecture.architecture_factory import get_architecture 
import shutil 
import glob  
import tensorflow as tf  
  

def save_weights_with_versioning(weights_dir, weights, max_versions=3):  
    """Save weights with versioning, keeping only the last max_versions"""  
    
      
    # Find existing weight files  
    existing_weights = glob.glob(f"{weights_dir}/weights_v*.pkl")  
    existing_weights.sort(key=lambda x: int(x.split('_v')[1].split('.')[0]))  
      
    # Remove old versions if we exceed max_versions  
    while len(existing_weights) >= max_versions:  
        os.remove(existing_weights.pop(0))  
      
    # Determine next version number  
    if existing_weights:  
        last_version = int(existing_weights[-1].split('_v')[1].split('.')[0])  
        next_version = last_version + 1  
    else:  
        next_version = 1  
      
    # Save new version  
    weights_path = f"{weights_dir}/weights_v{next_version}.pkl"  
    with open(weights_path, 'wb') as f:  
        pickle.dump(weights, f)  
      
    # Create/update symlink to latest  
    latest_path = f"{weights_dir}/weights"  
    if os.path.exists(latest_path):  
        os.remove(latest_path)  
    os.symlink(f"weights_v{next_version}.pkl", latest_path)  
    tf.keras.backend.clear_session()  

    return weights_path
    
class E2ESystem(Block):  
    """End-to-end system for neural receiver training"""  
      
    def __init__(self, config: ExperimentConfig, training=False):  
        super().__init__()  
        self.config = config  
        self._training = training  
  
        # Setup resource grid  
        self.resource_grid = ResourceGrid(  
            num_ofdm_symbols=config.ofdm.num_ofdm_symbols,  
            fft_size=config.ofdm.fft_size,  
            subcarrier_spacing=float(config.ofdm.subcarrier_spacing),  
            num_tx=1,  
            num_streams_per_tx=1,  
            cyclic_prefix_length=config.ofdm.cyclic_prefix_length,  
            dc_null=config.ofdm.dc_null,  
            pilot_pattern=config.ofdm.pilot_pattern,  
            pilot_ofdm_symbol_indices=config.ofdm.pilot_ofdm_symbol_indices,  
            num_guard_carriers=config.ofdm.num_guard_carriers  
        )  
  
        # Stream management  
        self.stream_manager = StreamManagement(np.array([[1]]), 1)  
  
        # Codeword length  
        self.n = int(self.resource_grid.num_data_symbols * config.modulation.num_bits_per_symbol)  
        self.k = int(self.n * config.modulation.coderate)  
  
        # Antenna configuration  
        self.ut_antenna = Antenna(  
            polarization="single",  
            polarization_type="V",  
            antenna_pattern="38.901",  
            carrier_frequency=float(config.channel.carrier_frequency)
        )  
  
        self.bs_array = AntennaArray(  
            num_rows=1,  
            num_cols=1,  
            polarization="dual",  
            polarization_type="VH",  
            antenna_pattern="38.901",  
            carrier_frequency=float(config.channel.carrier_frequency)
        )  
  
        ######################################  
        ## Transmitter  
        self._binary_source = BinarySource()  
        if not training:  
            self._encoder = LDPC5GEncoder(self.k, self.n)  
        self._mapper = Mapper("qam", config.modulation.num_bits_per_symbol)  
        self._rg_mapper = ResourceGridMapper(self.resource_grid)  
  
        ######################################  
        ## Channel  
        cdl = CDL(  
            config.channel.cdl_model,   
            float(config.channel.delay_spread),   
            float(config.channel.carrier_frequency),  
            self.ut_antenna,   
            self.bs_array,   
            "uplink",   
            min_speed=config.channel.speed  
        )  
        self._channel = OFDMChannel(cdl, self.resource_grid, normalize_channel=True, return_channel=True)  
  
        ######################################  
        ## Receiver  
        self._neural_receiver = get_architecture(  
            config.architecture.name,  
            num_bits_per_symbol=config.modulation.num_bits_per_symbol  # From modulation config  
        )
        
        self._rg_demapper = ResourceGridDemapper(self.resource_grid, self.stream_manager) 
        if not training:  
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
  
    @tf.function  
    def call(self, batch_size, ebno_db):  
        if len(ebno_db.shape) == 0:  
            ebno_db = tf.fill([batch_size], ebno_db)  
  
        ######################################  
        ## Transmitter  
        no = ebnodb2no(ebno_db, self.config.modulation.num_bits_per_symbol, self.config.modulation.coderate)  
        if self._training:  
            c = self._binary_source([batch_size, 1, 1, self.n])  
        else:  
            b = self._binary_source([batch_size, 1, 1, self.k])  
            c = self._encoder(b)  
          
        x = self._mapper(c)  
        x_rg = self._rg_mapper(x)  
  
        ######################################  
        ## Channel  
        no_ = expand_to_rank(no, tf.rank(x_rg))  
        y, h = self._channel(x_rg, no_)  
  
        ######################################  
        ## Receiver  
        y = tf.squeeze(y, axis=1)  
        llr = self._neural_receiver(y, no)  
        llr = insert_dims(llr, 2, 1)  
        llr = self._rg_demapper(llr)  
        llr = tf.reshape(llr, [batch_size, 1, 1, self.n])  
  
        if self._training:  
            # Compute BMD rate  
            bce = tf.nn.sigmoid_cross_entropy_with_logits(c, llr)  
            bce = tf.reduce_mean(bce)  
            rate = tf.constant(1.0, tf.float32) - bce/tf.math.log(2.)  
            return rate  
        else:  
            b_hat = self._decoder(llr)  
            return b, b_hat  
def train_neural_receiver(config_path: str = "Config/config.yaml"):    
    """Main training function"""    
    print("Loading configuration...")    
    config = load_config(config_path)    
      
    # Usar el valor de la configuration  
    resume_training = config.training.resume_training
      
    # Get modulation name  
    modulation_name = get_modulation_name(config.modulation.num_bits_per_symbol)  
      
    print(f"Starting training with architecture: {config.architecture.name}")  
    print(f"Using modulation: {modulation_name}")  
      
    # Create output directory with modulation  
    weights_dir = f"{config.paths.weights_dir}/{modulation_name}/{config.architecture.name}"  
    os.makedirs(weights_dir, exist_ok=True)  
       
    
    # Initialize model for training  
    model = E2ESystem(config, training=True)  
    # Load existing weights if resuming training    
    if resume_training and os.path.exists(f"{weights_dir}/weights"):    
        print("Loading existing weights to resume training...")    
        model(1, tf.constant(10.0, tf.float32))  # Build the model first    
        with open(f"{weights_dir}/weights", 'rb') as f:    
            weights = pickle.load(f)    
        for i, w in enumerate(weights):    
            model._neural_receiver.weights[i].assign(w)
            
    # Create optimizer based on config  
    if config.training.optimizer.lower() == "adam":  
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate)  
    else:  
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")  
  
    # Training loop  
    for i in range(config.training.num_training_iterations):  
        # Sample batch of SNRs  
        ebno_db = tf.random.uniform(  
            shape=[],   
            minval=config.snr.ebno_db_min,   
            maxval=config.snr.ebno_db_max  
        )  
          
        # Forward pass  
        with tf.GradientTape() as tape:  
            rate = model(config.training.batch_size, ebno_db)  
            loss = -rate  # Minimize negative rate  
          
        # Compute and apply gradients  
        weights = tape.watched_variables()  
        grads = tape.gradient(loss, weights)  
        optimizer.apply_gradients(zip(grads, weights))  
          
        # Print progress  
        if i % 250 == 0:  
            #print(f'Iteration {i}/{config.training.num_training_iterations}  Rate: {rate.numpy():.4f} bit', end='\r')  
            print(f'Iteration {i}/{config.training.num_training_iterations}  Rate: {rate.numpy():.4f} bit')  
              
            # Guardar checkpoint cada 100 iteraciones  
            if i > 0:  # No guardar en la primera iteración  
                checkpoint_weights = model._neural_receiver.weights  
                save_weights_with_versioning(weights_dir, checkpoint_weights)
    #weights_path = f"{weights_dir}/weights"  
    final_weights = model._neural_receiver.weights  
    save_weights_with_versioning(weights_dir, final_weights)
      
    # Save config used for this experiment  
    config_save_path = f"{weights_dir}/config.yaml"  
    from Utils.config_loader import save_config  
    save_config(config, config_save_path)  
      
    print(f"\nTraining completed! Weights saved with versioning in {weights_dir}")
    #print(f"Configuration saved to {config_save_path}")
  
if __name__ == "__main__":  
    import argparse  
    parser = argparse.ArgumentParser()  
    parser

#Use resume training from config    
#python main.py --mode train --config Config/config_neural_receiver1_qpsk.yaml  

#Forzar reentrenamiento
#python main.py --mode train --config Config/config_neural_receiver1_qpsk.yaml --no-resume

#testing
#python main.py --mode test --config Config/config_neural_receiver1_qpsk.yaml


