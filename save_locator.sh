#!/usr/bin/env bash
python3 Tools/freeze_graph.py --input_graph=Models/model_locator/1544520055/saved_model.pb --input_checkpoint=Models/model_locator/1544520055/variables/variables.data-00000-of-00001 --output_graph=Models/locator_model.pb --output_node_names=landmarks --input_binary=true