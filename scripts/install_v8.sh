WD=`pwd`
sudo apt-get install nodejs && \
npm install jsvu -g && \
export PATH="${HOME}/.jsvu:${PATH}" && \
jsvu --os=linux64 --engines=v8-debug
cd $WD
