def train_models(Xtr, ytr, batch_size = 200, num_inits=3, alpha0 = 1.0, num_iter = 100, eta = 0.2):
    #print("alpha0 = %0.2f" % alpha)
    yhat = np.zeros((num_inits, X_te_bin.shape[0]))
    all_params0 = []
    all_paramsf = []
    ytr_2d = jnp.reshape(ytr, (ytr.shape[0],1))
    #print(ytr_2d.shape)
    all_trains = []
    all_test = []
    alpha = alpha0/jnp.sqrt(N)
    for i in range(num_inits):
        #print("i = %d" % i)
        _, params0 = init_fn(random.PRNGKey(i), (-1,32,32,3)) # new init
        shift_apply = lambda params, Xin: alpha * ( apply_fn(params, Xin) - apply_fn(params0,Xin) )
        #log_soft = lambda params, X: log_softmax(shift_fn(params, X))
        loss_fn = lambda params, Xin, yin: jnp.mean( ( shift_apply(params, Xin) - yin )**2 )
        #loss_tr = lambda params: loss(params, Xtr, ytr)
        #grad_fn = grad(loss_fn, 0)
        grad_fn = jit(grad(loss_fn, 0))
        #print(loss_fn(params0, Xtr, ytr))
        #print(grad_fn(params0, Xtr, ytr))
        #loss = jit(loss)
        #accuracy = jit(accuracy)
        # sgd optimizer
        #eta_eff = eta / alpha**(1.5)
        #eta_eff = eta / alpha**(0.8)
        #const = 10.0**(1.5)
        #ratio_const = (10.0/alpha0)**(2.0)

        lr_exp = 1.8
        #lr_exp = 1.25
        #lr_exp = 0.25
        opt_init, opt_update, get_params = optimizers.momentum(N*eta/alpha0**(lr_exp),0.95)
        opt_state = opt_init(params0)
        #print(Xtr.shape)
        #print(ytr.shape)
        num_batches = Xtr.shape[0]//batch_size

        print("init %d" % i)
        for t in range(num_iter):
            loss_t = 0.0
            for b in range(num_batches):
            	opt_state = opt_update(t, grad_fn(get_params(opt_state),Xtr[b*batch_size:(b+1)*batch_size],ytr_2d[b*batch_size:(b+1)*batch_size]), opt_state)
            	loss_t += 1/num_batches * loss_fn(get_params(opt_state), Xtr[b*batch_size:(b+1)*batch_size],ytr_2d[b*batch_size:(b+1)*batch_size])
            sys.stdout.write('\r loss: %0.6f' % loss_t)
            if loss_t < 1e-2:
                break
        all_trains += [loss_t]
        all_params0 += [params0]
        all_paramsf += [get_params(opt_state)]
        tb = 100
        num_batch_te = X_te_bin.shape[0] // tb
        test_loss = 0.0
        for n in range(num_batch_te):
            yhat[i,tb*n:tb*(n+1)] = shift_apply(get_params(opt_state), X_te_bin[tb*n:tb*(n+1)])[:,0]
            test_loss += 1/num_batch_te * loss_fn(get_params(opt_state), X_te_bin[tb*n:(n+1)*tb],y_te_bin[tb*n:(n+1)*tb].reshape((tb,1)))
        all_test += [test_loss]
    test_ens = jnp.mean( (jnp.mean(yhat, axis = 0)-y_te_bin )**2  )
    np.save(savedir+"ens_test_loss_N={}_P={}_L={}_alpha0={:.2f}_logeta={:.2f}_sigma2=2".format(N,P, block_size, alpha0,np.log10(eta)), test_ens)
    np.save(savedir+"test_loss_N={}_P={}_L={}_alpha0={:.2f}_logeta={:.2f}_sigma2=2".format(N,P, block_size, alpha0,np.log10(eta)), all_test)
    np.save(savedir+"train_loss_N={}_P={}_L={}_alpha0={:.2f}_logeta={:.2f}_sigma2=2".format(N,P, block_size, alpha0,np.log10(eta)), all_trains)
    #np.save(modeldir+"params0_N={}_P={}_L={}_alpha0={:.2f}_logeta={}".format(N,P, block_size, alpha0,eta), all_params0)
    #np.save(modeldir+"paramsf_N={}_P={}_L={}_alpha0={:.2f}_logeta={}".format(N,P, block_size, alpha0,eta), all_paramsf)
    #np.save(savedir+"yhat_N={}_P={}_L={}_alpha0={:.2f}_logeta={}".format(N,P, block_size, alpha0,eta), yhat)
    return

np.save(savedir +"test_labels", y_te_bin)