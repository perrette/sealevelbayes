import pymc as pm

def getmodeldata(name, model=None):
    if model is None:
        model = pm.modelcontext(None)

    var = getattr(model, name)

    # e.g. pymc version 5.9.0
    if hasattr(var, "data"):
        return var.data

	# e.g. pymc version 5.13.1
    else:
        return var.eval()