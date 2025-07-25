# constants
ocean_surface_km2 = 361e6
water_density = 1000
ice_density = 916.8
km3_to_m3 = 1e9
m3_to_kg = ice_density
kg_to_gt = 1e-12
kg_to_mm_sle = 1e-3/water_density/ocean_surface_km2
gt_to_mm_sle = kg_to_mm_sle/kg_to_gt
km3_to_gt = ice_density * 1e-3
km3_to_mm_sle = km3_to_gt*gt_to_mm_sle