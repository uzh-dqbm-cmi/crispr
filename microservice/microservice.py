import os
import glob
import socket
import subprocess
import requests, json
from flask_cors import CORS
from flask import Flask, request, render_template, jsonify, send_file, after_this_request
from random import random
import os
import datetime
import numpy as np
import scipy
import pandas as pd
import torch
from torch import nn
import criscas
from criscas.utilities import create_directory, get_device, report_available_cuda_devices
from criscas.predict_model import *
from PyPDF2 import PdfFileMerger
import shutil
from threading import Timer

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["POST"])
def crispr():
    # POST
    if request.method == "POST":

        if not os.path.exists("temp_files"):
            os.makedirs("temp_files")

        # Get data and temporarily save as file
        file = request.files["data"]
        selectedBaseEditor = request.form[ "selectedBaseEditor" ]
        selectedPredictionType = request.form[ "selectedPredictionType" ]
        selectedSequences = request.form[ "selectedSequences" ].split(',')
        selectedAction = request.form["selectedAction"]
        print( selectedBaseEditor, selectedPredictionType, selectedSequences, selectedAction )

        if not file.filename == "":

            randomFolderName = str( int( random() * 100000 ) )

            base_dir = os.path.abspath(os.path.join(
                os.getcwd(),
                'temp_files'
            ))

            base_dir = create_directory(os.path.join(base_dir, randomFolderName))

            file.save( os.path.join(base_dir, 'uploadedData.csv' ) )

            seq_df = pd.read_csv( os.path.join( os.path.join('temp_files', randomFolderName, 'uploadedData.csv' ) ), header=0 )

            csv_dir = create_directory(os.path.join(base_dir, 'predictions'))

            device = get_device(False, 0)

            base_editor = selectedBaseEditor
            bedict = BEDICT_CriscasModel(base_editor, device)

            pred_w_attn_runs_df, proc_df = bedict.predict_from_dataframe(seq_df)

            pred_option = selectedPredictionType
            pred_w_attn_df = bedict.select_prediction(pred_w_attn_runs_df, pred_option)

            pred_w_attn_runs_df.to_csv(os.path.join(csv_dir, f'predictions_allruns.csv'))

            pred_w_attn_df.to_csv(os.path.join(csv_dir, f'predictions_predoption_{pred_option}.csv'))

            seqid_pos_map = {}
            for sequence in selectedSequences:
                seqid_pos_map[ sequence ] = []

            pred_option = selectedPredictionType
            apply_attn_filter = False
            fig_dir = create_directory( os.path.join( base_dir, 'fig_dir') )



            bedict.highlight_attn_per_seq(pred_w_attn_runs_df,
                                          proc_df,
                                          seqid_pos_map=seqid_pos_map,
                                          pred_option=pred_option,
                                          apply_attnscore_filter=apply_attn_filter,
                                          fig_dir=create_directory(os.path.join(fig_dir, pred_option)))




            if selectedAction == "plot":

                merger = PdfFileMerger()

                pdfNames = os.listdir(os.path.join(base_dir, 'fig_dir', selectedPredictionType))
                pdfs = []
                for pdfName in pdfNames:
                    pdfs.append(
                        os.path.join(
                            base_dir,
                            'fig_dir',
                            selectedPredictionType,
                            pdfName)
                    )

                print(len(pdfs))

                for pdf in pdfs:
                    merger.append(pdf)

                merger.write(
                    os.path.join(
                        base_dir,
                        'fig_dir',
                        selectedPredictionType,
                        'merged.pdf')
                )

                merger.close()

                @after_this_request
                def remove_file(response):
                    shutil.rmtree( base_dir )
                    return response

                return send_file(
                    os.path.join(
                        base_dir,
                        'fig_dir',
                        selectedPredictionType,
                        'merged.pdf'),
                    mimetype='application/pdf'
                )

            if selectedAction == "download":

                shutil.make_archive(
                    os.path.join(
                        base_dir,
                        'predictions_' + selectedBaseEditor + '_' + selectedPredictionType
                    ),
                    'zip',
                    os.path.join(
                        base_dir,
                        'predictions'
                    )
                )

                @after_this_request
                def remove_file(response):
                    shutil.rmtree( base_dir )
                    return response

                return send_file(
                    os.path.join(
                        base_dir,
                        'predictions_' + selectedBaseEditor + '_' + selectedPredictionType + '.zip'
                    ),
                    mimetype='application/pdf'
                )


        else:
            return jsonify(
                error='sth went wrong',
            )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4321, debug=True)
