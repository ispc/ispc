WD=`pwd`
sudo apt-get install nodejs && \
npm install jsvu -g && \
export PATH="${HOME}/.jsvu:${PATH}" && \
jsvu --os=linux64 v8-debug@10.2.154 && \
cp scripts/v8-redirect.sh ${HOME}/.jsvu/v8 && \
chmod +x ${HOME}/.jsvu/v8
cd $WD
