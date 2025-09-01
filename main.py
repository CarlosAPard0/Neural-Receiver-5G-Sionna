"""    
Main script to run training or testing of neural receiver    
"""    
import argparse    
from Utils.training import train_neural_receiver    
from Utils.testing import test_neural_receiver    
from Utils.config_loader import load_config, save_config  
    
def main():    
    parser = argparse.ArgumentParser(description='Neural Receiver Training and Testing')    
    parser.add_argument('--mode', choices=['train', 'test'], required=True,    
                        help='Mode: train or test')    
    parser.add_argument('--config', default='Config/config.yaml',    
                        help='Path to configuration file')    
    parser.add_argument('--resume', action='store_true',    
                        help='Force resume training (overrides config file)')    
    parser.add_argument('--no-resume', action='store_true',    
                        help='Force fresh training (overrides config file)')    
        
    args = parser.parse_args()    
      
    # Validar argumentos mutuamente excluyentes  
    if args.resume and args.no_resume:  
        parser.error("--resume and --no-resume cannot be used together")  
        
    if args.mode == 'train':    
        print("Starting training mode...")    
          
        # Si se especifica --resume o --no-resume, modificar la configuración temporalmente  
        if args.resume or args.no_resume:  
            config = load_config(args.config)  
            config.training.resume_training = args.resume  
            print(f"Overriding config: resume_training = {config.training.resume_training}")  
              
            # Guardar configuración temporal  
            temp_config_path = args.config.replace('.yaml', '_temp.yaml')  
            save_config(config, temp_config_path)  
            train_neural_receiver(temp_config_path)  
              
            # Limpiar archivo temporal  
            import os  
            os.remove(temp_config_path)  
        else:  
            # Usar configuración tal como está  
            train_neural_receiver(args.config)  
  
    elif args.mode == 'test':    
        print("Starting testing mode...")    
        test_neural_receiver(args.config)    
    
if __name__ == "__main__":    
    main()
    
    
    
    
    