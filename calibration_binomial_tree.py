import numpy as np
import pandas as pd

class CalibrationBinomialTree(object):
    
    def __init__(self, params):
        
        """
            The storage is implemented as an array of  
            size M=(N+1)(N+2)/2 where N is the number of    
            time steps. The i'th node at the k'th step 
            is indexed at k(k+1)/2 + i, i<=k.       

                                   14
                               9   
                            5      13
                         2     8  
                      0     4      12
                         1     7  
                            3      11
                               6   
                                   10
        """
        
        self.T = params["T"]
        self.N = params["N"]
        self.X0 = 0.0
        self.dt = self.T/self.N
        self.dx = np.sqrt(self.dt)
        self.M = (self.N+1)*(self.N+2)//2
        # xlist is a superset of x-grid points of all time steps
        self.xlist = self.X0 + self.dx*np.arange(-self.N,self.N+1)
        self.prob_tree = np.zeros(self.M)
        self.price_tree = np.zeros(self.M)
        self.set_marginal_distribution_type("S")
        
    def set_marginal_distribution_type(self, marginal_distribution_type):
        
        """ The markov functional mf(x) represents the asset price if the
            marginal_distribution_type is "S"; or of the logarithm of the 
            asset price if the marginal_distribution_type is LogS. """
        
        assert marginal_distribution_type in ("S", "LogS")
        self.marginal_distribution_type = marginal_distribution_type
        
    def forward_prop(self, mf):
        
        """ mf: Markov functional or its values on the grid of xlist """
        
        self.prob_tree.fill(0.0)
        self.prob_tree[0] = 1.0
        for k in range(self.N):
            n1 = k*(k+1)//2 # index base for step k
            n2 = (k+1)*(k+2)//2 # index base for step k+1
            nn1 = k+1 # number of nodes for step k
            pbase = self.prob_tree[n1:n1+nn1]
            ilist = np.arange(nn1)
            x_ind = self.N + 2*ilist - k
            
            if callable(mf):
                a = mf(self.xlist[x_ind+1])
                b = mf(self.xlist[x_ind-1])
                c = mf(self.xlist[x_ind])
            else:
                a = mf[x_ind+1]
                b = mf[x_ind-1] 
                c = mf[x_ind]
                
            if self.marginal_distribution_type == "LogS":
                a = np.exp(a)
                b = np.exp(b)
                c = np.exp(c)
                
            pu = 0.5*np.ones(nn1)
            pd = 0.5*np.ones(nn1)
            numer1 = c - b
            numer2 = a - c
            denom = a - b
            cc = (a-b>1e-9)
            pu[cc] = numer1[cc]/denom[cc]
            pd[cc] = numer2[cc]/denom[cc]
            self.prob_tree[n2:n2+nn1] += pbase*pd
            self.prob_tree[n2+1:n2+1+nn1] += pbase*pu
                        
    def price_forward(self, mf):
        
        probs = self.prob_tree[-(self.N+1):]
        ilist = np.arange(self.N+1)
        
        if callable(mf):
            ss = mf(self.xlist[2*ilist])
        else:
            ss = mf[2*ilist]
        
        if self.marginal_distribution_type == "LogS":
            ss = np.exp(ss)
        
        return np.sum(probs*ss)
    
    def price_vanilla(self, mf, opttype=1, r=0.0):
        
        assert opttype in (1,-1)
        
        probs = self.prob_tree[-(self.N+1):]
        ilist = np.arange(self.N+1)
        
        if callable(mf):
            Ks = mf(self.xlist[2*ilist])
        else:
            Ks = mf[2*ilist]
        
        if self.marginal_distribution_type == "LogS":
            Ks = np.exp(Ks)
        
        cum_p = np.cumsum(probs)
        cum_ps = np.cumsum(probs*Ks)
        prices = Ks*cum_p - cum_ps
        
        if opttype == 1:
            F0 = self.price_forward(mf)
            prices = prices + F0 - Ks
            
        prices *= np.exp(-r*self.T)
        return prices
    
    def price(self, payoff, mf, isAmerican=False, r=0.0, q=0.0):
                
        self.price_tree.fill(0.0)
        discount = np.exp(-r*self.dt)
        
        # terminal condition at time T, step N
        n1 = self.N*(self.N+1)//2
        nn1 = self.N+1
        for i in range(nn1):
            
            ilist = np.arange(nn1)
            
            if callable(mf):
                ss = mf(self.xlist[2*ilist])
            else:
                ss = mf[2*ilist]
            
            if self.marginal_distribution_type == "LogS":
                ss = np.exp(ss)
            
            self.price_tree[n1:] = payoff(ss)
            
        # back-propagation
        for k in range(self.N-1,-1,-1):
            
            n1 = k*(k+1)//2
            n2 = (k+1)*(k+2)//2
            nn1 = k+1
            ilist = np.arange(nn1)
            x_ind = self.N + 2*ilist - k
            
            if callable(mf):
                a = mf(self.xlist[x_ind+1])
                b = mf(self.xlist[x_ind-1])
                c = mf(self.xlist[x_ind])
            else:
                a = mf[x_ind+1]
                b = mf[x_ind-1]
                c = mf[x_ind]
            
            if self.marginal_distribution_type == "LogS":
                a = np.exp(a)
                b = np.exp(b)
                c = np.exp(c)
            
            pu = (c-b)/(a-b)
            pd = (a-c)/(a-b)
            self.price_tree[n1:n1+nn1] = discount*(pd*self.price_tree[n2:n2+nn1] + pu*self.price_tree[n2+1:n2+1+nn1])
            if isAmerican:
                exercise_values = payoff(c*np.exp(-(r-q)*(self.T-k*self.dt)))
                self.price_tree[n1:n1+nn1] = np.maximum(self.price_tree[n1:n1+nn1], exercise_values)
                
        return self.price_tree[0]
    
    def calibrate_European_T0T1(self, ppf_T1, maxiter=10):
        
        """ Time-homogeneous local volatility calibration. 
            ppf_T1 is the inverse CDF of asset price if marginal_distribution_type is bachelier, 
            or of the logarithm of the asset price if the marginal_distribution_type is black. """
        
        assert callable(ppf_T1)
        
        ilist = np.arange(self.N+1)
        mf = np.copy(self.xlist) # identity Markov functional
        xs = self.xlist[2*ilist]

        res = []
        
        niter = 0
        while niter < maxiter:            
            self.forward_prop(mf)
            probs = self.prob_tree[-(self.N+1):]
            cdf_xs = np.cumsum(probs)
            res.append({"xs": xs, "probs": probs, "cdf_xs": cdf_xs, "mf": mf})
            mf = np.interp(self.xlist, xp=xs, fp=ppf_T1(cdf_xs))
            niter += 1
            
        res.append({"xs": xs, "probs": probs, "cdf_xs": cdf_xs, "mf": mf})
        return res
    
    def evaluate_locvol_T0T1(self, mf, volatility_type="black"):
        
        """ Has to be called after calibrate_European_T0T1 """
        
        assert volatility_type in ("bachelier", "black")
        
        k = self.N-1
        n1 = k*(k+1)//2 # index base for step k
        n2 = (k+1)*(k+2)//2 # index base for step k+1
        nn1 = k+1 # number of nodes for step k
        ilist = np.arange(nn1)
        x_ind = self.N + 2*ilist - k
        
        if callable(mf):
            a = mf(self.xlist[x_ind+1])
            b = mf(self.xlist[x_ind-1])
            c = mf(self.xlist[x_ind])
        else:
            a = mf[x_ind+1]
            b = mf[x_ind-1] 
            c = mf[x_ind]
                        
        pu = 0.5*np.ones(nn1)
        pd = 0.5*np.ones(nn1)
        numer1 = c - b if self.marginal_distribution_type == "S" else np.exp(c)-np.exp(b)
        numer2 = a - c if self.marginal_distribution_type == "S" else np.exp(a)-np.exp(c)
        denom = a - b if self.marginal_distribution_type == "S" else np.exp(a)-np.exp(b)
        cc = (denom>1e-9)
        pu[cc] = numer1[cc]/denom[cc]
        pd[cc] = numer2[cc]/denom[cc]
        
        if self.marginal_distribution_type == "S":
            locvar = (pu*(a-c)**2 + pd*(b-c)**2)/self.dt
            if volatility_type == "black":
                locvar = locvar/c**2
        else:
            locvar = (pu*(a-c)**2 + pd*(b-c)**2)/self.dt
            if volatility_type == "bachelier":
                locvar = locvar*np.exp(2*c)
        #locvar = (pu*(np.log(a)-np.log(c))**2 + pd*(np.log(b)-np.log(c))**2)/self.dt
        xs = self.xlist[x_ind]
        ss = c if self.marginal_distribution_type == "S" else np.exp(c)

        return {"xs": xs, "ss": ss, "locvar": locvar}
      
