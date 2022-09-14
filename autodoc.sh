#!/usr/bin/env bash

# build the docs
cd documentation
make clean
make html
cd ..

# commit and push
git add documentation/*
git commit -m "building and pushing docs"
git push origin devel

# switch branches and pull the data we want
git checkout gh-pages
rm -rf *
touch .nojekyll
git checkout devel documentation/builddoc/html
mv ./documentation/builddoc/html/* ./
rm -rf ./documentation
git add -A
git commit -m "publishing updated docs..."
git push origin gh-pages

# switch back
git checkout devel