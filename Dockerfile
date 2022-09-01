FROM nginx:alpine

ADD nginx.conf /etc/nginx/conf.d/default.conf
ADD build/ /usr/share/nginx/html/
