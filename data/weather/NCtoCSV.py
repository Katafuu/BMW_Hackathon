import xarray as xr
DS = xr.open_dataset("./tas_1hr_HOSTRADA-v1-0_BE_gn_2024070100-2024073123.nc")
DS.to_dataframe().to_csv("august_airtempmean.csv")