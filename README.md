First steps:
1. Create your repository, say,
  https://gitlab.com/YOURUSERNAME/YOURREPO.git

2. mkdir a directory and clone your repo there:
```
$ mkdir workdir
$ cd workdir
$ git clone https://gitlab.com/YOURUSERNAME/YOURREPO.git
```

3. Add this repo as an upstream and fork to your repo:
$ git remote add upstream https://gitlab.com/wmatbd/ml.git
$ git fetch upstream
$ git checkout master
$ git merge upstream/master


