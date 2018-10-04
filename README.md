### First steps:
1. Create your repository, say,
```
  https://gitlab.com/YOURUSERNAME/YOURREPO.git
```

2. mkdir a directory and clone your (empty) repo there:
```
$ mkdir workdir
$ cd workdir
$ git clone https://gitlab.com/YOURUSERNAME/YOURREPO.git
$ cd YOURREPO
```

3. Add this repo as an upstream.
```
$ git remote add upstream https://gitlab.com/wmatbd/ml.git
```

### Update your repo to contain the changes from upstream:
```
$ git fetch upstream
$ git checkout master
$ git pull
```

### Work on your repo:
```
$ git add (files)
$ git commit -m "Your commit message"
$ git push origin master
```

