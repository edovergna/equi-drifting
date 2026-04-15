update_env:
	@echo "Updating snellius conda environment..."
	@echo "Generating environment.yaml from current environment..."
	bash generate_conda_environment.sh
	@echo "Launching snellius job to update environment..."
	sbatch jobs/install_env.sh