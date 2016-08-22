import globholder

def func2():
    print "swapping current glob"
    if globholder.current_glob == globholder.globs[0]:
        globholder.current_glob = globholder.globs[1]
    else:
        globholder.current_glob = globholder.globs[0]
        