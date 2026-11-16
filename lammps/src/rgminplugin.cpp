/* ----------------------------------------------------------------------
   Registers min_style rgmin with "plugin load".
   SPDX-License-Identifier: MIT
------------------------------------------------------------------------- */

#include "lammpsplugin.h"
#include "version.h"

#include "min_rgmin.h"

using namespace LAMMPS_NS;

static Min *min_rgmin_creator(LAMMPS *lmp)
{
  return new MinRgmin(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  auto register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "min";
  plugin.name = "rgmin";
  plugin.info = "rgmin minimizers (L-BFGS, FIRE, CG, ...) through the rgmin C ABI";
  plugin.author = "Rohit Goswami (rgoswami@ieee.org)";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &min_rgmin_creator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
