begin # Robustness test for the DeeoOpNet learn function
    NDIMS = [1,2,3]

    for i=1:length(NDIMS)
        print("-------------------------------\n\n")
        print("N-DIMS = $(NDIMS[i])")
        print("-------------------------------\n\n")
        # Number of dimnesions of the grid
        ndims = NDIMS[i]
        # Number of instances of the input a 
        na = 2
        # Number of grid points 
        res = 2 
        m = (2<<res)^ndims
        # Number of time points 
        nt = 2<<(res+1)
        # Total number of data points 
        ND = na * m * nt

        ## Munged Data
        # Input function a
        a = repeat(rand(m)',ND) # Each row is a data point
        # Output function 
        u = rand(ND)
        # Cogs
        inter = rand(ND,ndims+1) # One for x and the other for t )


        # NN Parameters 
        NL = 5
        DL = 10 
        p = 10
        lr = 0.01
        epochs = 2

        # Constructor
        dponet = DeepOpNet(a,inter,u,DL,NL,p,lr,epochs)
        model = learn(dponet)
    end 
end