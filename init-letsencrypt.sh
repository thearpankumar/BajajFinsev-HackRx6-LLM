#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

if [ ! -f docker-compose.yml ]; then
    echo "docker-compose.yml not found. Please run this script from the root of your project."
    exit 1
fi

# --- Configuration ---
domains=(llmnow.dev)
email="ms1725@srmist.edu.in" # IMPORTANT: Replace with your email
data_path="./data/certbot"
rsa_key_size=4096
# --- End Configuration ---

# Ensure data_path exists and has correct permissions
mkdir -p "$data_path/conf"
sudo chown -R $(whoami):$(whoami) "$data_path"

if [ -d "$data_path" ]; then
  read -p "Existing data found for $domains. Continue and replace existing certificate? (y/N) " decision
  if [ "$decision" != "Y" ] && [ "$decision" != "y" ]; then
    exit
  fi
fi

if [ ! -e "$data_path/conf/options-ssl-nginx.conf" ] || [ ! -e "$data_path/conf/ssl-dhparams.pem" ]; then
  echo "### Downloading recommended TLS parameters ..."
  curl -s https://raw.githubusercontent.com/certbot/certbot/master/certbot-nginx/certbot_nginx/_internal/tls_configs/options-ssl-nginx.conf > "$data_path/conf/options-ssl-nginx.conf"
  openssl dhparam -out "$data_path/conf/ssl-dhparams.pem" 2048
  echo
fi

echo "### Creating dummy certificate for $domains ..."
path="/etc/letsencrypt/live/$domains"
mkdir -p "$data_path/conf/live/$domains"

docker compose run --rm --entrypoint "\
  openssl req -x509 -nodes -newkey rsa:$rsa_key_size -days 1\
    -keyout '$path/privkey.pem' \
    -out '$path/fullchain.pem' \
    -subj '/CN=localhost'" certbot
echo

echo "### Starting Nginx ..."
docker compose up --force-recreate -d nginx
echo

echo "### Deleting dummy certificate and any old certificates for $domains ..."
docker compose run --rm --entrypoint "\
  rm -rf /etc/letsencrypt/live/$domains* && \
  rm -rf /etc/letsencrypt/archive/$domains* && \
  rm -rf /etc/letsencrypt/renewal/$domains*.conf" certbot
echo

echo "### Requesting Let's Encrypt certificate for $domains ..."
domain_args=""
for domain in "${domains[@]}"; do
  domain_args="$domain_args -d $domain"
done

case "$email" in
  "") email_arg="--register-unsafely-without-email" ;;
  *) email_arg="--email $email" ;;
esac

docker compose run --rm --entrypoint "\
  certbot certonly --webroot -w /var/www/certbot \
    $email_arg \
    $domain_args \
    --rsa-key-size $rsa_key_size \
    --agree-tos \
    --force-renewal" certbot
echo

# --- Robustness Check ---
# Add a check to ensure the certificate was actually created before reloading nginx.
if [ ! -f "$data_path/conf/live/$domains/fullchain.pem" ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "ERROR: Certificate generation FAILED."
    echo "The certbot command finished, but the certificate file was not found."
    echo "Check the logs above for errors from Let's Encrypt."
    echo "Common causes:"
    echo "  - The domain name ($domains) does not point to this server's IP."
    echo "  - A firewall is blocking port 80."
    echo "  - Let's Encrypt rate limits have been exceeded."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 1
fi

echo "### Reloading Nginx ..."
docker compose exec nginx nginx -s reload

echo
echo "-------------------------------------------------"
echo "SUCCESS! Your certificate has been generated."
echo "You can now start your full stack with:"
echo "docker compose up -d"
echo "-------------------------------------------------"
