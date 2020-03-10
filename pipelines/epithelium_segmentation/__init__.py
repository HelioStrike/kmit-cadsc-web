from PIL import Image

def preprocess(x):
    pass

def run_model(x):
    pass

def postprocess(x):
    pass

def get_display_image(fname):
    inp = Image.open(fname)
    preprocessed = preprocess(inp)
    pred = run_model(preprocessed)
    postprocessed = postprocess(pred)
