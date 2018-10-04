First steps:
1. Create your repository, say,
```
  https://gitlab.com/YOURUSERNAME/YOURREPO.git
```

2. mkdir a directory and clone your repo there:
```
$ mkdir workdir
$ cd workdir
$ git clone https://gitlab.com/YOURUSERNAME/YOURREPO.git
```

3. Add this repo as an upstream.
```
$ git remote add upstream https://gitlab.com/wmatbd/ml.git
```

4. Update your repo to contain the changes from upstream:
```
$ git fetch upstream
$ git checkout master
$ git pull
```

5. Work on your repo:
```
$ git add (files)
$ git commit -m "Your commit message"
$ git push origin master
```

