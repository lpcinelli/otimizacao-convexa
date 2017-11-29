import numpy as np
import sympy as sp

def deriveFunction(symbolVec, costSymbolic, args):
    # Derives from the given symbolic function, the desired numeric functions such as
    # the func itself, its gradient, hessian and etc

    funcs = {}
    if 'func' in args:
        cost = sp.lambdify(symbolVec, costSymbolic(symbolVec), modules=['numpy'])
        funcs['func'] = cost
    if 'grad' in args:
        gradCost = sp.lambdify(symbolVec, [sp.diff(costSymbolic(symbolVec), var) \
                                            for var in symbolVec], 'numpy')
        funcs['grad'] = gradCost
    if 'hess' in args:
        hessianCost = sp.lambdify(symbolVec, \
                                    sp.hessian(costSymbolic(symbolVec), symbolVec), 'numpy')
        funcs['hess'] = hessianCost
    if 'jaco' in args:
        jacobian = sp.lambdify(symbolVec, costSymbolic(symbolVec).jacobian(symbolVec), \
                                modules=['numpy'])
        funcs['jaco'] = jacobian

    return funcs

def generalDescentMethod(x0, costSymbolic, specificDescentMethod, eps=1e-6, **args):

    symbolVec = sp.symbols('a0:%d'%len(x0))
    x0 = x0.squeeze()
    descentAlgo = specificDescentMethod(symbolVec, costSymbolic, **args)
    nbEvalDict = {}
    params = {}
    delta = np.inf
    xHist = [x0]
    fxHist = [np.NaN]

    while np.linalg.norm(delta) > eps:

        step, fx, partialNbEvalDict, params, delta = descentAlgo(x0, params)
        nbEvalDict = {k: nbEvalDict.get(k, 0) + \
                        partialNbEvalDict.get(k, 0) for k in set(partialNbEvalDict)}

        x0 = x0 + step
        xHist.append(x0)
        fxHist.append(fx)

    return xHist, fxHist, nbEvalDict

def steepestDescent(symbolVec, costSymbolic, **args):

    costFuncs = deriveFunction(symbolVec, costSymbolic, ['func', 'grad'])
    cost = costFuncs['func']
    gradCost = costFuncs['grad']
    lineSearchMethod = args.pop('lineSearchMethod', noLineSearch)

    def steepestDescentAlgo(x0, params):

        dfx = np.asarray(gradCost(*x0))
        direction = -dfx
        t, f, nbEval, newParams = lineSearchMethod(x0, dfx, direction, cost, params, **args)
        step = t*direction

        return step, f, {'nbFuncEval': nbEval, 'nbGradEval':1}, newParams, step

    return steepestDescentAlgo

def newtonDescent(symbolVec, costSymbolic, **args):

    costFuncs = deriveFunction(symbolVec, costSymbolic, ['func', 'grad', 'hess'])
    cost = costFuncs['func']
    gradCost = costFuncs['grad']
    hessianCost = costFuncs['hess']
    lineSearchMethod = args.pop('lineSearchMethod', noLineSearch)

    def basicNewtonAlgo(x0, params):

        dfx = np.asarray(gradCost(*x0))
        Hx = np.asarray(hessianCost(*x0))
        eigVals = np.linalg.eigvals(Hx)
        if np.any(eigVals < 0.1):
            beta = 1.5*np.abs(np.min(eigVals))
            Hx = (Hx + beta*np.eye(Hx.shape[0]))/(1+beta)

        HxInv = np.linalg.inv(Hx)
        direction = -HxInv.dot(dfx)
        t, f, nbEval, newParams = lineSearchMethod(x0, dfx, direction, cost, params, **args)
        step = t*direction

        return step, f, {'nbFunEval': nbEval, 'nbGradEval':1, 'nbHessEval':1}, newParams, step

    return basicNewtonAlgo

def gaussNewtonDescent(symbolVec, costSymbolic, **args):

    costFuncs = deriveFunction(symbolVec, costSymbolic, ['func', 'jaco'])
    cost = costFuncs['func']
    jacobian = costFuncs['jaco']
    scalarCost = sp.lambdify(symbolVec, \
                            costSymbolic(symbolVec).dot(costSymbolic(symbolVec)), modules=['numpy'])
    lineSearchMethod = args.pop('lineSearchMethod', noLineSearch)

    def gaussNewtonAlgo(x0, params):

        fxScalarPrevious = params.pop('Fx', np.inf)

        fx = cost(*x0)
        Jx = jacobian(*x0)
        dfx = 2*Jx.T.dot(fx)
        Hx = 2*Jx.T.dot(Jx)

        L, D = hessianCorrectionMatDav(Hx)
        Dinv = np.linalg.inv(D)
        y = -L.dot(dfx)
        direction = L.T.dot((Dinv).dot(y))
        direction = direction.reshape((np.max(direction.shape),))

        t, f, nbEval, newParams = lineSearchMethod(x0, dfx, direction, scalarCost, params, **args)
        step = t*direction

        fxScalar = scalarCost(*(x0+step))

        return step, fxScalar, {'nbFunEval': nbEval, 'nbGradEval': dfx.shape[0]}, \
                dict(**newParams, Fx=fxScalar), np.abs(fxScalar-fxScalarPrevious)

    return gaussNewtonAlgo

def hessianCorrectionMatDav(H):

    n = H.shape[1]
    L = np.zeros(H.shape)
    D = np.zeros(n)
    h00 = 0

    if H[0, 0] > 0:
        h00 = H[0, 0]
    else:
        h00 = 1

    for k in range(1, n):
        m = k - 1
        L[m, m] = 1
        if H[m, m] <= 0:
            H[m, m] = h00

        for i in range(k, n):
            L[i, m] = - H[i, m]/H[m, m]
            H[i, m] = 0
            for j in range(k, n):
                H[i, j] += L[i, m]*H[m, j]

        if H[k, k] > 0 and H[k, k] < h00:
            h00 = H[k, k]

    L[-1, -1] = 1
    if H[-1, -1] <= 0:
        H[-1, -1] = h00
    for i in range(n):
        D[i] = H[i, i]

    return L, np.diag(D)

def noLineSearch(x, dfx, vec, cost, params, **args):

    alphaHat = params.pop('alpha', 1)
    fxk = params.pop('fxk', None)
    nbEval = 0
    if fxk is None:
        fxk = cost(*x)
        nbEval += 1

    fHat = cost(*(x + alphaHat*vec))
    dfx2alpha = np.vdot(dfx, dfx)*alphaHat
    alpha = (dfx2alpha*alphaHat)/(2*(fHat - fxk + dfx2alpha))
    fx = cost(*(x + alpha*vec))

    return alpha, fx, nbEval+2, {'alpha': alpha, 'fxk': fx}

def backtrackingLineSearch(x, dfx, vec, cost, params, **args):

    domain = args.pop('domain', None)
    fx = args.pop('fx', None)
    alpha = args.pop('alpha', 0.15)
    beta = args.pop('beta', 0.5)

    t = 1
    nbEval = 1

    if fx is None:
        fx = cost(*x)
        nbEval += 1

    if domain is not None:
        while (x + t*vec <= domain[0]) or (domain[1] <= x + t*vec):
            ptiny
            t *= beta

    f = cost(*(x + t*vec))

    while f > (fx + alpha*t*np.vdot(dfx, vec)):
        t *= beta
        f = cost(*(x + t*vec))
        nbEval += 1

    return t, f, nbEval, {}

def conjugateMethod(symbolVec, costSymbolic, **args):

    costFuncs = deriveFunction(sp.Matrix(symbolVec), costSymbolic, ['func', 'grad', 'hess'])
    cost = costFuncs['func']
    gradCost = costFuncs['grad']
    hessianCost = costFuncs['hess']

    lineSearchMethod = args.pop('lineSearchMethod', None)
    if lineSearchMethod is None or lineSearchMethod is noLineSearch:
        def conjugateGradient(x0, params):

            prevDirection = params.pop('prevDirection', np.zeros(x0.shape))
            prevGrad = params.pop('prevGrad', None)
            dfx = np.asarray(gradCost(*x0)).squeeze()
            Hx = np.asarray(hessianCost(*x0)).squeeze()

            if prevGrad is None:
                beta = 0
            else:
                beta = np.vdot(dfx, dfx)/np.vdot(prevGrad, prevGrad)

            direction = -dfx + beta*prevDirection.squeeze()
            alpha = np.vdot(dfx, dfx)/(np.dot(direction, np.dot(Hx, direction)))
            step = alpha*direction
            f = np.asarray(cost(*(x0 + step)))

            return step, f, {'nbFunEval': 1, 'nbGradEval':1, 'nbHessEval':1}, \
                    {'prevDirection': direction, 'prevGrad': dfx}, step

        return conjugateGradient

    elif lineSearchMethod is backtrackingLineSearch:
        def fletcherReeves(x0, params):

            k = params.pop('k', 0) + 1
            prevDirection = params.pop('prevDirection', None)
            prevGrad = params.pop('prevGrad', None)
            dfx = np.asarray(gradCost(*x0)).squeeze()

            if prevGrad is None:
                beta = 0
                prevDirection = np.zeros(x0.shape)
            else:
                beta = np.vdot(dfx, dfx)/np.vdot(prevGrad, prevGrad)

            direction = -dfx + beta*prevDirection.squeeze()
            alpha, f, nbEval, newParams = lineSearchMethod(x0, dfx, direction, \
                                                           cost, params, **args)
            step = alpha*direction

            if k == x0.size:
                direction = None
                dfx = None
                k = 0

            return step, f, {'nbFunEval': nbEval, 'nbGradEval':1}, \
                    {**newParams, 'prevDirection': direction, 'prevGrad': dfx, 'k': k}, step

        return fletcherReeves

    else:
        raise Exception('Invalid line search method')
