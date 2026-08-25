def builder(x):
    recipe = compute(x)
    return {"a": recipe.value}

def dumper(y):
    # the exact shape that reached the Docker gate: `recipe` belongs to
    # builder(), not here
    return {"b": recipe.stop_epoch}

def compute(x):
    return x

def closure_ok(items):
    total = len(items)
    def inner(v):
        return v + total          # legitimate closure -- must NOT be flagged
    return [inner(i) for i in items]

def lambda_ok(rows):
    return sorted(rows, key=lambda r: r[1])   # lambda arg -- must NOT be flagged
