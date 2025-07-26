#!/bin/bash

set -e

DOMAIN="llmnew.dev"
EMAIL="admin@llmnew.dev"

echo "Setting up SSL certificates for $DOMAIN..."

if ! command -v certbot &> /dev/null; then
    echo "Installing certbot..."
    sudo apt update
    sudo apt install -y certbot python3-certbot-nginx
fi

sudo systemctl stop nginx 2>/dev/null || true

echo "Obtaining SSL certificate..."
sudo certbot certonly \
    --standalone \
    --email $EMAIL \
    --agree-tos \
    --no-eff-email \
    --domains $DOMAIN,www.$DOMAIN

echo "Setting up auto-renewal..."
sudo crontab -l 2>/dev/null | grep -v certbot || true
(sudo crontab -l 2>/dev/null; echo "0 12 * * * /usr/bin/certbot renew --quiet --post-hook 'systemctl reload nginx'") | sudo crontab -

sudo mkdir -p /etc/nginx/sites-available
sudo mkdir -p /etc/nginx/sites-enabled
sudo mkdir -p /var/www/certbot

sudo cp nginx/sites-available/llmnew.dev.conf /etc/nginx/sites-available/
sudo ln -sf /etc/nginx/sites-available/llmnew.dev.conf /etc/nginx/sites-enabled/

sudo nginx -t

sudo systemctl start nginx
sudo systemctl enable nginx

echo "SSL setup complete!"
echo "Your API will be available at: https://$DOMAIN"
echo "API documentation: https://$DOMAIN/docs"
