WD=`pwd`
sudo apt install nodejs && \
npm install jsvu -g && \
export PATH="${HOME}/.jsvu:${PATH}" && \
jsvu --os=linux64 --engines=v8-debug
cd $WD
