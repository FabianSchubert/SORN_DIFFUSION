import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2.
mpl.rcParams['font.size'] = 8.8
mpl.rcParams['legend.fontsize'] = 8.8
#mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['axes.color_cycle'] = ['009BDE', 'FF8800', '00EB8D', 'FBC15E', '8EBA42', 'FFB5B8']
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['patch.facecolor'] = '009BDE'
mpl.rcParams['mathtext.fontset'] = 'stixsans'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.family'] = 'sans-serif'

mpl.rcParams['svg.fonttype'] = 'none'

#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['mathtext.default'] = 'sf'

default_fig_width = 5.26874

file_format = [".png",".svg",".eps"] ## cycle through file formats and append to string when saving plots

# base folder of simulation data (adjust for your local system)

sim_data_base_folder = '/media/fabian/linux_drive2/sim_data/'
plots_base_folder = '/home/fabian/Uni/Neuro/plots/'
sim_data_base_folder = '/media/fabian/linux_drive2/sim_data/'
plots_base_folder = '/home/fabian/Uni/Neuro/plots/'
