def clean_pattern(x, pattern):
    x = str(x)
    match = re.search(pattern, x)
    if match:
        weight = match.group(1)# or match.group(3)
        weight_value = float(weight)
        return weight_value
    else:
        return ''