import os
map = {}

currdir = "/Users/madhuseelam/sre/sauron"
envs = os.listdir(currdir + "/envs")
for dir in envs:
    print (dir)
    if (os.path.isdir(currdir + "/envs/" + dir)):
        services = os.listdir(currdir + "/envs" + "/" + dir)
        newservices = []
        for i in services:
            if (os.path.exists(currdir + "/envs/" + dir + "/" + i + "/group_vars/all/all.yml")):
                newservices.append(i)
        if (len(newservices) > 0):
            map[dir] = newservices


map