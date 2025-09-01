#!/bin/bash
set -e

for file in Config/*.yaml; do
    sed -i 's/num_training_iterations: 15000/num_training_iterations: 50/' "$file"
done

echo "✅ Todos los archivos YAML han sido modificados."
