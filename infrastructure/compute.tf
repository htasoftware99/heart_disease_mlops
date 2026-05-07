data "google_compute_image" "ubuntu_2404" {
  family  = "ubuntu-2404-lts-amd64"
  project = "ubuntu-os-cloud"
}

resource "google_compute_instance" "gitops_vm" {
  name         = "gitops-machine"
  machine_type = "e2-standard-4" # 4 vCPU, 16 GB Memory
  zone         = "us-central1-a"

  # (IP Forwarding - Networking)
  can_ip_forward = true

  # (HTTP, HTTPS ve Load Balancer Health Check)
  tags = ["http-server", "https-server", "allow-health-check"]

  # OS and Disk Settings (Ubuntu 20.04 LTS and 128 GB Balanced PD)
  boot_disk {
    initialize_params {
      image = data.google_compute_image.ubuntu_2404.self_link
      size  = 128
      type  = "pd-balanced"
    }
  }

  # Network Settings
  network_interface {
    network = "default"

    # The access_config block assigns an Ephemeral Public IP address to the machine so that it can be accessed from outside.
    access_config {
      // Ephemeral IP
    }
  }

  # Attaching the service account that created in iam.tf to the VM.
  service_account {
    email  = google_service_account.heart_disease.email
    scopes = ["cloud-platform"]
  }
}

# Firewall rule required for Load Balancer Health Check.
resource "google_compute_firewall" "allow_health_checks" {
  name    = "allow-health-checks"
  network = "default"

  allow {
    protocol = "tcp"
  }

  # Google Cloud Load Balancer IP interval
  source_ranges = ["130.211.0.0/22", "35.191.0.0/16"]
  target_tags   = ["allow-health-check"]
}