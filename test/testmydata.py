import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import sys
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def slope_fit(x, y, low, high):

    poly_output = np.polyfit(x[low:high], y[low:high], 1, full = True)
    pfit_info   = poly_output[0]
    poly_resid  = poly_output[1][0]
    pfit_slope  = pfit_info[0]
    pfit_c      = pfit_info[1]

    return pfit_slope, pfit_c, poly_resid



if __name__ == "__main__":

	iters = sys.argv[1]
	num_t_steps = int(iters)
	

	# #--------------------------------------------------------
	# ## Vorticity
	# #--------------------------------------------------------
	# with h5.File("./test/MyTestData/StructureFunctions/Vort_StrFunc_Data{}.h5".format(iters), "r") as in_f:

	# 	Dsets      = list(in_f.keys())
	# 	num_pow    = len(Dsets)
	# 	dimx, dimy = in_f[Dsets[0]][:, :].shape
	# 	Nx, Ny     = dimx*2, dimy*2
	# 	vort_str_func = np.zeros((num_pow, dimx, dimy))
	# 	for p, dset in enumerate(Dsets):
	# 		vort_str_func[p, :, :] = in_f[dset][:, :]

	# # 2D Imshow of Vorticity SF
	# fig = plt.figure(figsize = (16, 8))
	# gs  = GridSpec(1, 1)
	# ax1 = fig.add_subplot(gs[0, 0])
	# im1 = ax1.imshow(vort_str_func[1, :, :], extent = (0.0, 0.5, 0.5, 0.0), cmap = "jet")
	# ax1.set_xlabel(r"$r_x$")
	# ax1.set_ylabel(r"$r_y$")
	# ax1.set_xlim(0.0, 0.5)
	# ax1.set_ylim(0.0, 0.5)
	# div1  = make_axes_locatable(ax1)
	# cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
	# cb1   = plt.colorbar(im1, cax = cbax1)
	# cb1.set_label(r"$\mathcal{S}_2^{\omega}(r)$")
	# ax1.set_title(r"2nd Order Vorticity Structure Function")
	# plt.savefig("./test/MyTestData/StructureFunctions/" + "Vorticity_SF_2D_Imshow.png")
	# plt.close()


	# Max_vort_r = int(np.ceil(np.sqrt((Nx - 1)**2 + (Ny - 1)**2)))
	# r          = np.zeros(Max_vort_r)
	# for i in range(Max_vort_r):
	# 	r_x  = i / (Nx - 1)
	# 	r_y  = i / (Ny - 1)
	# 	r[i] = np.sqrt(2) * i / (2 * Max_vort_r)
	# r2 = r**2
	# r3 = r**3

	# # Shell average over vorticity increments
	# shell_avg_vort_str_func = np.zeros((num_pow, Max_vort_r))
	# shell_counts            = np.zeros((Max_vort_r, ))
	# for i in range(vort_str_func.shape[1]):
	# 	for j in range(vort_str_func.shape[2]):
	# 		# Get shell index
	# 		r_ind = int(np.ceil(np.sqrt(i**2 + j**2)))

	# 		for p in range(num_pow):
	# 			shell_avg_vort_str_func[p, r_ind] += vort_str_func[p, i, j] / num_t_steps

	# 		# Update count
	# 		shell_counts[r_ind] += 1

	# # Normalize shell average
	# for p in range(num_pow):
	# 	shell_avg_vort_str_func[p, :] /= shell_counts[:]

	# ## Combined plot
	# fig = plt.figure(figsize = (12, 8))
	# gs  = GridSpec(1, 1)
	# # 2nd order
	# ax1 = fig.add_subplot(gs[0, 0])
	# # ax1.plot(r[:Max_vort_r], r2[:Max_vort_r], 'k--', label = r"$r^2$")
	# for i in range(num_pow):
	# 	ax1.plot(r[:Max_vort_r], shell_avg_vort_str_func[i, :], label = r"$\mathcal{S}^{\omega}_p(r)$;" + "$p = {}$".format(i + 1))
	# ax1.set_xlabel(r"$r$")
	# ax1.set_ylabel(r"$\mathcal{S}^{\omega}_p$")
	# ax1.set_yscale('log')
	# ax1.set_xscale('log')
	# ax1.legend()
	# plt.savefig("./test/MyTestData/StructureFunctions/" + "Vorticity_SF_1D_All.png")
	# plt.close()



	#--------------------------------------------------------
	## Longitudinal Velocity
	#--------------------------------------------------------
	with h5.File("./test/MyTestData/StructureFunctions/Vel_LongStrFunc_Data{}.h5".format(iters), "r") as in_f:

		Dsets      = list(in_f.keys())
		num_pow    = len(Dsets)
		dimx, dimy = in_f[Dsets[0]][:, :].shape
		Nx, Ny     = dimx*2, dimy*2
		vel_long_str_func = np.zeros((num_pow, dimx, dimy))
		for p, dset in enumerate(Dsets):
			vel_long_str_func[p, :, :] = in_f[dset][:, :]

	# 2D Imshow of vel_longicity SF
	fig = plt.figure(figsize = (16, 8))
	gs  = GridSpec(1, 1)
	ax1 = fig.add_subplot(gs[0, 0])
	im1 = ax1.imshow(vel_long_str_func[1, :, :], extent = (0.0, 0.5, 0.5, 0.0), cmap = "jet")
	ax1.set_xlabel(r"$r_x$")
	ax1.set_ylabel(r"$r_y$")
	ax1.set_xlim(0.0, 0.5)
	ax1.set_ylim(0.0, 0.5)
	div1  = make_axes_locatable(ax1)
	cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
	cb1   = plt.colorbar(im1, cax = cbax1)
	cb1.set_label(r"$\mathcal{S}_2^{\mathbf{u}_{\parallel}}(r)$")
	ax1.set_title(r"2nd Order vel_longicity Structure Function")
	plt.savefig("./test/MyTestData/StructureFunctions/" + "vel_long_SF_2D_Imshow.png")
	plt.close()


	Max_vel_long_r = int(np.ceil(np.sqrt((Nx - 1)**2 + (Ny - 1)**2)))
	r          = np.zeros(Max_vel_long_r)
	for i in range(Max_vel_long_r):
		r_x  = i / (Nx - 1)
		r_y  = i / (Ny - 1)
		r[i] = np.sqrt(2) * i / (2 * Max_vel_long_r)
	r2 = r**2
	r3 = r**3

	# Shell average over vel_longicity increments
	shell_avg_vel_long_str_func = np.zeros((num_pow, Max_vel_long_r))
	shell_counts            = np.zeros((Max_vel_long_r, ))
	for i in range(vel_long_str_func.shape[1]):
		for j in range(vel_long_str_func.shape[2]):
			# Get shell index
			r_ind = int(np.ceil(np.sqrt(i**2 + j**2)))

			for p in range(num_pow):
				shell_avg_vel_long_str_func[p, r_ind] += vel_long_str_func[p, i, j] / num_t_steps

			# Update count
			shell_counts[r_ind] += 1

	# Normalize shell average
	for p in range(num_pow):
		shell_avg_vel_long_str_func[p, :] /= shell_counts[:]

	## Combined plot
	fig = plt.figure(figsize = (12, 8))
	gs  = GridSpec(1, 1)
	# 2nd order
	ax1 = fig.add_subplot(gs[0, 0])
	# ax1.plot(r[:Max_vel_long_r], r2[:Max_vel_long_r], 'k--', label = r"$r^2$")
	for i in range(num_pow):
		ax1.plot(r[:Max_vel_long_r], np.absolute(shell_avg_vel_long_str_func[i, :]), label = r"$\mathcal{S}^{\mathbf{u}_{\parallel}}_p(r)$;" + "$p = {}$".format(i + 1))
	ax1.set_xlabel(r"$r$")
	ax1.set_ylabel(r"$|\mathcal{S}^{\mathbf{u}_{\parallel}}_p|$")
	ax1.set_yscale('log')
	ax1.set_xscale('log')
	ax1.legend()
	plt.savefig("./test/MyTestData/StructureFunctions/" + "vel_long_SF_1D_All.png")
	plt.close()


	#--------------------------------------------------------
	## Transverse Velocity
	#--------------------------------------------------------
	with h5.File("./test/MyTestData/StructureFunctions/Vel_TransStrFunc_Data{}.h5".format(iters), "r") as in_f:

		Dsets      = list(in_f.keys())
		num_pow    = len(Dsets)
		dimx, dimy = in_f[Dsets[0]][:, :].shape
		Nx, Ny     = dimx*2, dimy*2
		vel_trans_str_func = np.zeros((num_pow, dimx, dimy))
		for p, dset in enumerate(Dsets):
			vel_trans_str_func[p, :, :] = in_f[dset][:, :]

	# 2D Imshow of vel_transicity SF
	fig = plt.figure(figsize = (16, 8))
	gs  = GridSpec(1, 1)
	ax1 = fig.add_subplot(gs[0, 0])
	im1 = ax1.imshow(vel_trans_str_func[1, :, :], extent = (0.0, 0.5, 0.5, 0.0), cmap = "jet")
	ax1.set_xlabel(r"$r_x$")
	ax1.set_ylabel(r"$r_y$")
	ax1.set_xlim(0.0, 0.5)
	ax1.set_ylim(0.0, 0.5)
	div1  = make_axes_locatable(ax1)
	cbax1 = div1.append_axes("right", size = "10%", pad = 0.05)
	cb1   = plt.colorbar(im1, cax = cbax1)
	cb1.set_label(r"$\mathcal{S}_2^{\mathbf{u}_{\perp}}(r)$")
	ax1.set_title(r"2nd Order vel_trans Structure Function")
	plt.savefig("./test/MyTestData/StructureFunctions/" + "vel_trans_SF_2D_Imshow.png")
	plt.close()


	Max_vel_trans_r = int(np.ceil(np.sqrt((Nx - 1)**2 + (Ny - 1)**2)))
	r          = np.zeros(Max_vel_trans_r)
	for i in range(Max_vel_trans_r):
		r_x  = i / (Nx - 1)
		r_y  = i / (Ny - 1)
		r[i] = np.sqrt(2) * i / (2 * Max_vel_trans_r)
	r2 = r**2
	r3 = r**3

	# Shell average over vel_trans increments
	shell_avg_vel_trans_str_func = np.zeros((num_pow, Max_vel_trans_r))
	shell_counts            = np.zeros((Max_vel_trans_r, ))
	for i in range(vel_trans_str_func.shape[1]):
		for j in range(vel_trans_str_func.shape[2]):
			# Get shell index
			r_ind = int(np.ceil(np.sqrt(i**2 + j**2)))

			for p in range(num_pow):
				shell_avg_vel_trans_str_func[p, r_ind] += vel_trans_str_func[p, i, j] / num_t_steps

			# Update count
			shell_counts[r_ind] += 1

	# Normalize shell average
	for p in range(num_pow):
		shell_avg_vel_trans_str_func[p, :] /= shell_counts[:]

	## Combined plot
	fig = plt.figure(figsize = (12, 8))
	gs  = GridSpec(1, 1)
	# 2nd order
	ax1 = fig.add_subplot(gs[0, 0])
	# ax1.plot(r[:Max_vel_trans_r], r2[:Max_vel_trans_r], 'k--', label = r"$r^2$")
	for i in range(num_pow):
		ax1.plot(r[:Max_vel_trans_r], shell_avg_vel_trans_str_func[i, :], label = r"$\mathcal{S}^{\mathbf{u}_{\perp}}_p(r)$;" + "$p = {}$".format(i + 1))
	ax1.set_xlabel(r"$r$")
	ax1.set_ylabel(r"$\mathcal{S}^{\mathbf{u}_{\perp}}_p$")
	ax1.set_yscale('log')
	ax1.set_xscale('log')
	ax1.legend()
	plt.savefig("./test/MyTestData/StructureFunctions/" + "Vel_trans_SF_1D_All.png")
	plt.close()