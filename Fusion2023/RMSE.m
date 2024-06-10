function metric=RMSE(x_p,x_q)
metric=sqrt(mean((x_p-x_q).^2));
end