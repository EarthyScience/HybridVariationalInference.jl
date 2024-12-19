tmpf = (a=1:5) -> begin
    Main.@infiltrate_main
    a .* 21
end
tmpf()