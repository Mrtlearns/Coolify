# flyimg — Coolify deployment notes

## Purpose
This folder (`/xyz/flyimg`) contains the Docker Compose deployment for the Flyimg service used by Coolify.

## Files in this folder
- `docker-compose.yml` — primary Compose file (single source of truth).
- `docker-compose.override.yml` — optional overrides (volume mounts, secret mounts, custom nginx/php config).
- `nginx-config/flyimage.conf` — custom nginx site fragment (example includes `client_max_body_size`).
- `config/parameters.yml` — Flyimg application config (mounts over `/var/www/html/config/parameters.yml` inside container).

> NOTE: Docker Compose will normally merge `docker-compose.override.yml` with `docker-compose.yml` when both are present. We rely on this behavior to keep overrides separate from the base file.

---

## Coolify UI settings (exact values to use)

When creating the resource in Coolify:

1. **Create New Resource → Git Based → Public Repository** (your repo is public).
2. **Build Pack**: **Docker Compose**.  
   Coolify will use your Compose file as the source of truth for volumes, env, ports, etc. :contentReference[oaicite:0]{index=0}
3. **Base Directory**: `/Coolify`  
   (Set this to the repository folder that contains the `xyz` folder. If your repo layout differs, use the folder that is the parent of `/xyz`.)
4. **Docker Compose Location**: `/xyz/flyimg/docker-compose.yml`  
   (This path is combined with the Base Directory; make sure extension matches exactly.) :contentReference[oaicite:1]{index=1}

---

## What the override file does
- `docker-compose.override.yml` is used to add or override mounts/services defined in `docker-compose.yml` (for example mounting `config/parameters.yml` and `nginx-config/flyimage.conf`). Place the override file **in the same folder** as `docker-compose.yml` so it is discovered/merged by Docker Compose. :contentReference[oaicite:2]{index=2}

---

## Known Coolify caveats & suggested fallback workflows
- **Override files sometimes ignored**: there are community reports/issues where Coolify ignored `docker-compose.override.yml` (see linked issue). If your override mounts do not appear inside the container after deploy, use the fallback below. :contentReference[oaicite:3]{index=3}  
- **Repo files may not always be copied**: older issues report that Coolify sometimes only copies the Compose file and not all referenced repo files/folders. If you see empty folders created but missing file contents, try the fallback. :contentReference[oaicite:4]{index=4}

**Fallback options (if override not applied):**
1. Copy the `volumes:` entries from `docker-compose.override.yml` into `docker-compose.yml` (merge override into base). This is the most reliable workaround.
2. Use Coolify **Raw Compose** mode and paste the merged compose yaml there.
3. Put the required files (nginx conf, parameters.yml) in the same folder as the compose file so Coolify’s helper sees them.

---

## Verification steps (after deploy)

1. Confirm Coolify deployed the correct files to the application working folder:
   - On the Coolify host (or using Coolify terminal), check the app folder:
     ```bash
     ls -la /data/coolify/applications/<your_app_folder>
     ```
     Confirm `docker-compose.yml`, `docker-compose.override.yml`, `nginx-config/`, and `config/` appear.

2. Confirm container mounts and files:
   ```bash
   docker ps                      # find the flyimg container name or id
   docker exec -it flyimg /bin/sh   # or /bin/bash if available
   ls -l /etc/nginx/conf.d/
   cat /var/www/html/config/parameters.yml
   nginx -t                       # validate config if nginx is present
