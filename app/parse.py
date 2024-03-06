
def get_first_params(text):
    """
    
    """
    args = text.split('):')[0].split('(')[-1]
    args = [s.strip() for s in args.split(',')] if ',' in args else [args]
    return args[0].split(":")[0] if ":" in args[0] else args[0]
