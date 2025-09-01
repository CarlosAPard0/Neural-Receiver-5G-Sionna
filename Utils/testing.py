import os  
import tensorflow as tf  
import pickle  
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
from datetime import datetime  
  
from sionna.phy.utils import sim_ber  
from Utils.training import E2ESystem  
from Utils.config_loader import load_config, get_modulation_name


  
def test_neural_receiver(config_path: str = "Config/config.yaml"):    
    """Test the trained neural receiver with all available checkpoints"""    
    print("Testing neural receiver...")    
        
    # Load configuration    
    config = load_config(config_path)    
        
    # Get modulation name    
    modulation_name = get_modulation_name(config.modulation.num_bits_per_symbol)    
        
    # Base weights directory  
    weights_base_dir = f"{config.paths.weights_dir}/{modulation_name}/{config.architecture.name}"  
      
    # Find all versioned weight files  
    import glob  
    weight_files = glob.glob(f"{weights_base_dir}/weights_v*.pkl")  
    weight_files.sort(key=lambda x: int(x.split('_v')[1].split('.')[0]))  
      
    if not weight_files:  
        print("No weight files found!")  
        return  
      
    # Create base results directory  
    date_str = datetime.now().strftime("%Y-%m-%d")    
    #base_results_dir = f"{config.paths.results_dir}/{modulation_name}/{date_str}"  
    base_results_dir = f"{config.paths.results_dir}/{modulation_name}/{config.architecture.name}/{date_str}"

      
    # SNR range for evaluation from config    
    ebno_range = config.evaluation.ebno_range    
    ebno_dbs = np.arange(ebno_range[0], ebno_range[1], ebno_range[2])    
      
    all_results = {}  
      
    # Test each checkpoint  
    for weight_file in weight_files:  
        # Extract version number for folder naming  
        version_num = weight_file.split('_v')[1].split('.')[0]  
        checkpoint_name = f"checkpoint_v{version_num}"  
          
        print(f"\nTesting {checkpoint_name}...")  
          
        # Create specific results directory for this checkpoint  
        results_dir = f"{base_results_dir}/{checkpoint_name}"  
        os.makedirs(results_dir, exist_ok=True)    
        os.makedirs(f"{results_dir}/plots", exist_ok=True)  
          
        # Initialize model for evaluation    
        model = E2ESystem(config, training=False)    
        model(1, tf.constant(10.0, tf.float32))  # Build the model  
          
        # Load specific checkpoint weights  
        with open(weight_file, 'rb') as f:    
            weights = pickle.load(f)    
            
        for i, w in enumerate(weights):    
            model._neural_receiver.weights[i].assign(w)    
            
        # Run evaluation    
        print(f"Running BER/BLER simulation for {checkpoint_name}...")    
        ber, bler = sim_ber(model, ebno_dbs,     
                          batch_size=config.evaluation.batch_size,     
                          num_target_block_errors=config.evaluation.num_target_block_errors,     
                          max_mc_iter=config.evaluation.max_mc_iter)    
            
        # Save results to CSV    
        results_df = pd.DataFrame({    
            'EbNo_dB': ebno_dbs,    
            'BER': ber.numpy(),  
            'BLER': bler.numpy()    
        })    
        csv_path = f"{results_dir}/{config.architecture.name}_{checkpoint_name}_results.csv"    
        results_df.to_csv(csv_path, index=False)    
        print(f"Results saved to {csv_path}")    
          
        # Store results for comparison plot  
        all_results[checkpoint_name] = {  
            'ber': ber.numpy(),  
            'bler': bler.numpy()  
        }  
          
        # Create individual plots for this checkpoint  
        # BLER plot  
        plt.figure(figsize=(10, 6))    
        plt.semilogy(ebno_dbs, bler.numpy(), 'o-', label=f'{config.architecture.name} {checkpoint_name}')    
        plt.xlabel('Eb/No [dB]')    
        plt.ylabel('BLER')    
        plt.title(f'{config.architecture.name} BLER Performance - {checkpoint_name}')    
        plt.grid(True)    
        plt.legend()    
        bler_plot_path = f"{results_dir}/plots/{config.architecture.name}_{checkpoint_name}_bler.png"    
        plt.savefig(bler_plot_path, dpi=300, bbox_inches='tight')    
        plt.close()  # Close to free memory  
          
        # BER plot  
        plt.figure(figsize=(10, 6))    
        plt.semilogy(ebno_dbs, ber.numpy(), 's-', label=f'{config.architecture.name} {checkpoint_name}', color='red')    
        plt.xlabel('Eb/No [dB]')    
        plt.ylabel('BER')    
        plt.title(f'{config.architecture.name} BER Performance - {checkpoint_name}')    
        plt.grid(True)    
        plt.legend()    
        ber_plot_path = f"{results_dir}/plots/{config.architecture.name}_{checkpoint_name}_ber.png"    
        plt.savefig(ber_plot_path, dpi=300, bbox_inches='tight')    
        plt.close()  # Close to free memory  
          
        print(f"Plots saved to {results_dir}/plots/")  
      
    # Create comparison plots with all checkpoints  
    comparison_dir = f"{base_results_dir}/comparison"  
    os.makedirs(comparison_dir, exist_ok=True)  
    os.makedirs(f"{comparison_dir}/plots", exist_ok=True)  
      
    # BLER comparison plot  
    plt.figure(figsize=(12, 8))  
    for checkpoint_name, results in all_results.items():  
        plt.semilogy(ebno_dbs, results['bler'], 'o-', label=f'{checkpoint_name}')  
    plt.xlabel('Eb/No [dB]')  
    plt.ylabel('BLER')  
    plt.title(f'{config.architecture.name} BLER Comparison - All Checkpoints')  
    plt.grid(True)  
    plt.legend()  
    comparison_bler_path = f"{comparison_dir}/plots/bler_comparison.png"  
    plt.savefig(comparison_bler_path, dpi=300, bbox_inches='tight')  
    plt.close()  
      
    # BER comparison plot  
    plt.figure(figsize=(12, 8))  
    for checkpoint_name, results in all_results.items():  
        plt.semilogy(ebno_dbs, results['ber'], 's-', label=f'{checkpoint_name}')  
    plt.xlabel('Eb/No [dB]')  
    plt.ylabel('BER')  
    plt.title(f'{config.architecture.name} BER Comparison - All Checkpoints')  
    plt.grid(True)  
    plt.legend()  
    comparison_ber_path = f"{comparison_dir}/plots/ber_comparison.png"  
    plt.savefig(comparison_ber_path, dpi=300, bbox_inches='tight')  
    plt.close()  
      
    print(f"\nComparison plots saved to {comparison_dir}/plots/")  
    print(f"Testing completed for all {len(weight_files)} checkpoints!")  
      
    return all_results
  
if __name__ == "__main__":  
    test_neural_receiver()