# machines.yaml hardware fields

`central/config/machines.yaml` accepts optional hardware fields that override agent discovery.

```yaml
machines:
  - machine_id: gb10
    label: "Dell Pro Max (GB10)"
    agent_base_url: "http://10.0.0.5:9001"
    cpu_cores: 64
    cpu_physical_cores: 32
    total_system_ram_bytes: 549755813888
    gpu:
      name: "NVIDIA GB10"
      type: "discrete"            # discrete | unified
      vram_bytes: 68719476736     # 64 GiB
      cuda_compute: [12, 1]
      driver_version: "545.101"
      pci_bus: "0000:01:00.0"
```

If fields are present, central uses them for model-fit calculations and reporting. Agent-side
discovery still runs, but machine entries in `machines.yaml` take precedence.
