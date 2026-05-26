import os
import sys
import base64
import pika
import pickle

import src.Model
import src.Log
from ultralytics import YOLO

class Server:
    def __init__(self, config):
        # RabbitMQ
        self.address = config["rabbit"]["address"]
        self.username = config["rabbit"]["username"]
        self.password = config["rabbit"]["password"]
        self.virtual_host = config["rabbit"]["virtual-host"]

        self.model_name = config["server"]["model"]
        self.total_clients = config["server"]["clients"]
        self.cut_layer = config["server"]["cut-layer"]
        self.batch_size = config["server"]["batch-size"]

        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, 5672, f'{self.virtual_host}', credentials))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')

        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.list_clients = []
        self.count_clients = 0

        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        self.data = config["data"]
        self.compress  = dict(config["compress"])
        self.max_frames = int(config.get("max_frames", 0))
        self.dry_run   = bool(config.get("cpcdrl", {}).get("dry_run", False))

        log_path = config["log-path"]
        self.logger = src.Log.Logger(f"{log_path}/app.log" , config["debug-mode"])
        self.logger.log_info(f"Application start. Server is waiting for {self.total_clients} clients.")
        src.Log.print_with_color(f"Application start. Server is waiting for {self.total_clients} clients.", "green")

        self.controllers: dict = {}   # device_name -> CpcdrlController
        self.client_device: dict = {} # str(client_id) -> edge device name
        self._init_controller(config)

    def _init_controller(self, config: dict) -> None:
        """Initialise one CPCDRL controller per edge device; observe last-session feedback."""
        cpcdrl_cfg = config.get("cpcdrl", {})
        if not cpcdrl_cfg.get("enable", False):
            return

        try:
            from src.CpcdrlController import CpcdrlController
            from src.CpcdrlAdapter import (
                normalize_device_name, get_device_timing, get_device_bandwidth_mbps,
                make_real_profile, controller_save_path, feedback_file_path, read_feedback,
            )

            raw_edge  = cpcdrl_cfg.get("edge_device",  "DEVICE_1")
            raw_cloud = cpcdrl_cfg.get("cloud_device", "DAI")

            edge_devices = ([normalize_device_name(raw_edge)]
                            if isinstance(raw_edge, str)
                            else [normalize_device_name(d) for d in raw_edge])
            cloud_device = normalize_device_name(
                raw_cloud[0] if isinstance(raw_cloud, list) else raw_cloud
            )

            src.Log.print_with_color(
                f"[CPCDRL] Edge devices: {edge_devices}  Cloud: {cloud_device}", "green"
            )

            for edge_device in edge_devices:
                save_path = controller_save_path(edge_device)
                if os.path.exists(save_path):
                    ctrl = CpcdrlController.load(save_path)
                    if ctrl.edge_device != edge_device or ctrl.cloud_device != cloud_device:
                        src.Log.print_with_color(
                            f"[CPCDRL] {edge_device}: device changed, updating profile.", "yellow"
                        )
                        ctrl.edge_device   = edge_device
                        ctrl.cloud_device  = cloud_device
                        ctrl.edge_timing   = get_device_timing(edge_device)
                        ctrl.cloud_timing  = get_device_timing(cloud_device)
                        ctrl.bandwidth_mbps = get_device_bandwidth_mbps(edge_device)
                        ctrl.profile = make_real_profile(
                            edge_device    = edge_device,
                            model_key      = ctrl.model_key,
                            bandwidth_mbps = ctrl.bandwidth_mbps,
                        )
                    src.Log.print_with_color(
                        f"[CPCDRL] Controller for {edge_device} loaded.", "green"
                    )
                else:
                    ctrl = CpcdrlController(
                        edge_device  = edge_device,
                        cloud_device = cloud_device,
                        model_name   = self.model_name,
                        batch_size   = self.batch_size,
                    )
                    src.Log.print_with_color(
                        f"[CPCDRL] Fresh controller for {edge_device}.", "green"
                    )

                feedback = read_feedback(edge_device)
                if feedback:
                    src.Log.print_with_color(
                        f"[CPCDRL] {edge_device} last session — "
                        f"e2e={feedback['e2e_latency_ms']:.1f} ms  "
                        f"mAP@50={feedback['map50']:.4f}", "green"
                    )
                    ctrl.observe(feedback["e2e_latency_ms"], feedback["map50"])
                    ctrl.save()
                    try:
                        os.remove(feedback_file_path(edge_device))
                    except OSError:
                        pass

                self.controllers[edge_device] = ctrl

        except Exception as exc:
            src.Log.print_with_color(f"[CPCDRL] Controller init failed: {exc}", "red")
            self.controllers = {}

    def on_request(self, ch, method, _, body):
        message = pickle.loads(body)
        action = message["action"]

        if action == "REGISTER":
            client_id = message["client_id"]
            layer_id = message["layer_id"]
            edge_device = message.get("edge_device")

            if (str(client_id), layer_id) not in self.list_clients:
                self.list_clients.append((str(client_id), layer_id))

            if layer_id == 1 and edge_device:
                self.client_device[str(client_id)] = edge_device

            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            self.register_clients[layer_id-1] += 1

            if self.register_clients == self.total_clients:
                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                self.notify_clients()

        elif action == "NOTIFY":
            self.count_clients +=1
            if self.count_clients == self.total_clients[1]:
                self.logger.log_info("Stop Inference !!!")
                self.notify_clients(start=False)
                sys.exit()
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def send_to_response(self, client_id, message):
        reply_queue_name = f"reply_{client_id}"
        self.reply_channel.queue_declare(reply_queue_name, durable=False)
        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        self.reply_channel.basic_publish(
            exchange='',
            routing_key=reply_queue_name,
            body=message
        )

    def start(self):
        self.channel.start_consuming()

    def notify_clients(self, start=True):
        if start:
            default_splits = {"a": 4, "b": 11, "c": 17, "d": 23}

            if os.path.exists(f"{self.model_name}.pt"):
                src.Log.print_with_color(f"Exist {self.model_name}", "green")
            else:
                src.Log.print_with_color(f"Download {self.model_name}", "yellow")
                _ = YOLO(f"{self.model_name}.pt")

            if os.path.exists(f"{self.model_name}.pt"):
                src.Log.print_with_color(f"Send model {self.model_name} to devices.", "green")
                with open(f"{self.model_name}.pt", "rb") as f:
                    encoded = base64.b64encode(f.read()).decode('utf-8')
            else:
                src.Log.print_with_color(f"{self.model_name} does not exist.", "yellow")
                sys.exit()

            for (client_id, layer_id) in self.list_clients:
                if layer_id == 1:  # edge — decide per-device cut point
                    device = self.client_device.get(client_id)
                    ctrl   = self.controllers.get(device) if device else None
                    if ctrl is not None:
                        try:
                            cut_int, bits_per_layer = ctrl.decide()
                            splits   = cut_int
                            compress = dict(self.compress)
                            compress["num_bits_per_layer"] = bits_per_layer
                            compress["num_bit"] = bits_per_layer[-1] if bits_per_layer else 8
                            ctrl.save()
                            src.Log.print_with_color(
                                f"[CPCDRL] {device} -> splits={splits}  "
                                f"bits/layer={bits_per_layer}", "green"
                            )
                        except Exception as exc:
                            src.Log.print_with_color(
                                f"[CPCDRL] decide() failed for {device} ({exc}), "
                                f"using defaults.", "yellow"
                            )
                            splits   = default_splits[self.cut_layer]
                            compress = dict(self.compress)
                    else:
                        splits   = default_splits[self.cut_layer]
                        compress = dict(self.compress)
                else:  # cloud — loads full model; splits value unused for slicing
                    splits   = 0
                    compress = dict(self.compress)

                compress["max_frames"] = self.max_frames
                compress["dry_run"]   = self.dry_run
                response = {
                    "action":     "START",
                    "message":    "Server accept the connection",
                    "model":      encoded,
                    "splits":     splits,
                    "batch_size": self.batch_size,
                    "num_layers": len(self.total_clients),
                    "model_name": self.model_name,
                    "data":       self.data,
                    "compress":   compress,
                }
                self.send_to_response(client_id, pickle.dumps(response))
        else:
            response = {"action": "STOP", "message": "Stop inference !!!"}
            for (client_id, layer_id) in self.list_clients:
                self.send_to_response(client_id, pickle.dumps(response))
