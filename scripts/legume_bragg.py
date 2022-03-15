import legume
import matplotlib.pyplot as plt
D = 0.6
r = 0.3
epsr = 12.0

lattice = legume.Lattice('square')
phc = legume.PhotCryst(lattice)
phc.add_layer(d=0.5, eps_b=1.0)
phc.add_layer(d=0.5, eps_b=epsr)
#phc.layers[-1].add_shape(legume.Circle(eps=1.0, r=r))
gme = legume.GuidedModeExp(phc, gmax=1)

legume.viz.structure(phc, xz=True, yz=True, figsize=3.)

path = lattice.bz_path(['G', 'M'], [15, ])
gme.run(kpoints=path['kpoints'],
        #gmode_inds=[0, 3],
        numeig=2,
        verbose=False)
fig, ax = plt.subplots(1, figsize = (7, 5))
legume.viz.bands(gme, figsize=(5,5), Q=False, ax=ax)
ax.set_xticks(path['indexes'])
ax.set_xticklabels(path['labels'])
ax.xaxis.grid('True')
#plt.figure()
#[ plt.plot(gme.kpoints[0, :], gme.freqs[:, i], 'r.') for i in range(2) ]
plt.show()