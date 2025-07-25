import matplotlib as mpl
from matplotlib.font_manager import FontProperties

mpl.rcParams.update({
    'axes.titleweight': 'bold',
    'axes.titlesize': 9,
    'legend.title_fontsize': 9  # for size
    # Note: font weight for legend title must be handled via fontproperties
})

legend_title_font = FontProperties(weight='bold', size=9)

sourcecolors = {
#     "total"
    # "steric": "orange",
    "steric": "wheat",
    "glacier": "mediumseagreen",
    "gis": "lightblue",
    "ais": "lavender",
    # "landwater": "rosybrown",
    "landwater": "darkcyan",
#     "vlm": "tab:brown",
    "vlm": "brown",
    "gia": "brown",
    # "vlm_res": "palevioletred",
    "vlm_res": "rosybrown",
    # "vlm_res": "lightgray",
    # "total": "tab:blue",
    "total": "steelblue",
    # "total": "navy",
}

sourcelabels = {
    "steric": "Steric",
    "glacier": "Glaciers",
    "gis": "Greenland",
    "ais": "Antarctica",
    "landwater": "Landwater",
    "gia": "GIA",
    "vlm_res": "VLM$_{res}$",
    "vlm": "VLM",
    "total": "Total",
    "gmsl": "GMSL",
}

sourcelabelslocal = sourcelabels.copy()
sourcelabelslocal.update({
    "steric": "Sterodynamic",
})

xcolors = {
    # 'ssp585': 'tab:orange',
    # # 'ssp585': 'steelblue',
    # 'ssp126': 'tab:green',
    # 'historical': 'black',
    'ssp585': (152/255, 0, 2/255),
    'ssp126': (0, 52/255, 102/255),
    # 'historical': 'black',
    "rcp26": (0, 52/255, 102/255),
    "rcp45": "tab:orange",
    "rcp85": (152/255, 0, 2/255),

    "C1": "#9acfe1",
    "C2": "#76855f",
    "C3": "#717c97",
    "C4": "#a9c57e",
    "C5": "#8ba7cc",
    "C6": "#f9c281",
    "C7": "#ef8870",
    "C8": "#bb715e",

    "difference": "black",
    "difference (GS)": "gray",
    "difference (SP)": "black",
    # "church2011": "gray",
    # "dangendorf2019": "darkgray",
    "church2011": "tab:purple",
    "dangendorf2019": "tab:orange",
}


xcolors['CurPol'] = xcolors['C7']
xcolors['GS'] = xcolors['C3']
xcolors['SP'] = xcolors['C1']
# xcolors['difference (GS)'] = xcolors['GS']
# xcolors['difference (SP)'] = xcolors['SP']


xlabels = {
    'ssp126': 'SSP1-2.6',
    'ssp370': 'SSP3-7.0',
    'ssp585': 'SSP5-8.5',
    'ssp126_mu': 'SSP1-2.6',
    'ssp585_mu': 'SSP5-8.5',
    'isimip_ssp126': 'SSP1-2.6',
    'isimip_ssp370': 'SSP3-7.0',
    'isimip_ssp585': 'SSP5-8.5',
    'historical': 'Hindcast',
    # 'historical': 'Historical',
    'SP': 'Sustainable Development',
    'GS': 'Gradual Strengthening',
    'CurPol': 'Current Policies',
    "C1": "C1 (< 1.5°C)",
    "C2": "C2 (< 1.5°C overshoot)",
    "C3": "C3 (< 2°C)",
    "C4": "C4 (< 2°C 67%)",
    "C5": "C5 (< 2.5°C)",
    "C6": "C6 (< 3°C)",
    "C7": "C7 (< 4°C)",
    "C8": "C8 (> 4°C)",
    "difference": "mitigation potential",
    "difference (GS)": "mitigation potential (GS)",
    "difference (SP)": "mitigation potential (SP)",

}


basincolors = {
    'Indian Ocean - South Pacific': "#fee624",
    'Northwest Pacific': "#8fd547",
    'East Pacific': "#37b779",
    'South Atlantic': "#2b8a88",
    'Subtropical North Atlantic': "#31698b",
    'Subpolar North Atlantic': "#450055",
    'Subpolar North Atl. West': "#433882",
    'Subpolar North Atl. East': "#450055",
    'Northwest Atlantic': "#31698b",
    'Northeast Atlantic': "#450055",
    'Mediterranean': "tab:orange",
}



fieldlabels = {
    "rsl": "Relative Sea Level",
    "rad": "Vertical Land Motion",
    "gsl": "Geocentric Sea Level",
}

diaglabels = {
    "slr20": '1901-1990',
    "past20c": '1901-1990',
    "rate2000": '1993-2018',
    "proj2050": '2050',
    "proj2100": '2100',
}