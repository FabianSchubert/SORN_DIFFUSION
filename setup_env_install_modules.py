import os

sim_data_base_folder = str(raw_input("Location of base folder for simulation data:"))

plots_base_folder = str(raw_input("Location of base folder for plots:"))

with open("./custom_modules/custom_modules/plot_setting.py","a") as setting_file:
	setting_file.write("sim_data_base_folder = '" + sim_data_base_folder + "'\n")
	setting_file.write("plots_base_folder = '" + plots_base_folder + "'\n")

os.system("pip install --user ./custom_modules/")
