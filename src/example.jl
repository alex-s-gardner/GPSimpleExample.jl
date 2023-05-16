using Arrow
using DataFrames
using CairoMakie
using GaussianProcesses
using Statistics
using GaussianProcesses
using Interpolations
using GPSimpleExample

dhdt_file = "src/data/lat[+50+52]lon[-126-124].cop30_v2"
grid_file = "src/data/lat[+50+52]lon[-126-124].arrow"

# # read in dhdt data
# df = DataFrame(Arrow.Table(dhdt_file))
# df = hcat(df, DataFrame(Arrow.Table(dhdt_file*"+")))
# df = delete!(copy(df), .!(df.landice))

# # local X/Y
# df[!, :X], df[!, :Y] = GPSimpleExample.epsg2epsg(copy(df.longitude), copy(df.latitude), "EPSG:4326", "EPSG:32609", parse_output=true)

# # subset
# df = df[1:5:end, :]; 

# # save local copy
# Arrow.write(dhdt_file*"_small", df::DataFrame)

# read data
df = DataFrame(Arrow.Table(dhdt_file * "_small"))

# bin as a funciton of elevation 
x_binned, y_binned, bin_count = GPSimpleExample.bin(Float64.(df.h), copy(df.trend), 0:100:3000, method=median)

# detrend data
itp = linear_interpolation(Float64.(x_binned), vec(y_binned),  extrapolation_bc = NaN)
df[!, :trend_anom] = df.trend .- itp(Float64.(df.h))

#Select mean and covariance function
mZero = MeanZero()
kern = Matern(5/2,[log(1e5), log(1e5), log(200)], 0.1) #Matern 5/2 ARD kernel
idx = .!isnan.(df.trend_anom)

X = vcat(df.X[idx]', df.Y[idx]', df.h[idx]')
dhdt_anom_foo = df.trend_anom[idx]'
df[!, :X], df[!, :Y] = GPSimpleExample.epsg2epsg(copy(df.longitude), copy(df.latitude), "EPSG:4326", "EPSG:32609", parse_output=true)


@time gp = GP(X, vec(Float64.(dhdt_anom_foo)), mZero, kern, log(0.24))
Xu1 = Matrix(quantile(X[1,:], [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,0.65, 0.7, 0.98])')
Xu2 = Matrix(quantile(X[2,:], [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,0.65, 0.7, 0.98])')
Xu3 = Matrix(quantile(X[3,:], [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,0.65, 0.7, 0.98])')

Xu = vcat(Xu1, Xu2, Xu3)

@time gp = GaussianProcesses.DTC(X, Xu, vec(dhdt_anom_foo), mZero, kern, log(0.24))

# # load grid data
# df = DataFrame(Arrow.Table(grid_file))
# df = delete!(copy(df), .!(df.landice))
# df[!, :X], df[!, :Y] = GPSimpleExample.epsg2epsg(df.lon, df.lat, "EPSG:4326", "EPSG:32609", parse_output=true)
# Arrow.write(grid_file * "_small", df::DataFrame)

# read grid data
df = DataFrame(Arrow.Table(grid_file * "_small"))


Xp = vcat(df.X', df.Y', df.h')
@time Yp, Sp = predict_y(gp, Xp)

pred = (Yp .+ itp(Float64.(df.h)));

colorrange = (-5, 5);
cmap = :balance

fig = Figure()
scatter(df.X, df.Y; color=pred, colormap=cmap, markersize=1, strokewidth=0, colorrange=colorrange)