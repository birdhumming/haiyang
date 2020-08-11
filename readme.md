- google site security prevented wget to download files, must make it public first! (very hard to figure out)
- wget needs options to add .html to filename:

wget --no-parent \        
       --recursive \
       --page-requisites \
       --html-extension \
       --base="https://sites.google.com/iu.edu/programming" https://sites.google.com/iu.edu/programming


- google site downloaded files still have auto redirect to google site itself; haven't figured out yet
4nosite line long

- web browsers are caching files from previous visit, so can't see current changes in html code

- github must commit file to add to online - quite inconvenient vs google
