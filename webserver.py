from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, HTTPServer
from Ember3D import *

aa_list = list("ACDEFGHIKLMNPQRSTVWY")

parser = ArgumentParser()
parser.add_argument('-p', '--port', default='24398', dest="port", type=int)
parser.add_argument('-d', '--device', default='cuda:0', dest="device", type=str)
parser.add_argument('-m', '--model', default="model/EMBER3D.model", dest='model_checkpoint', type=str)
parser.add_argument('--t5-dir', dest='t5_dir', default="./ProtT5-XL-U50/", type=str)
args = parser.parse_args()

landing_page = """<html>
<head>
<script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
<style>
.mol-container {
  width: 60%;
  height: 400px;
  position: relative;
}
</style>
</head>
<body>
<div align="center">
<p><textarea id="sequence_box", rows="4", cols="160"></textarea></p>
<p><button onclick="showStructure()">Submit</button></p>
<script>
function showStructure() {
  viewer.clear()
  var seq = document.getElementById("sequence_box").value
  let pdbUri = 'https://ember3d.rostlab.org/?' + seq;
  jQuery.ajax( pdbUri, {
    success: function(data) {
      let v = viewer;
      v.addModel( data, "pdb" );                       /* load data */
      v.setStyle({}, {cartoon: {colorscheme:{prop:'b',gradient: new $3Dmol.Gradient.ROYGB(0,100)}}});
      v.zoomTo();                                      /* set camera */
      v.render();                                      /* render scene */
      v.zoom(1.2, 1000);                               /* slight zoom */
    },
    error: function(hdr, status, err) {
      console.error( "Failed to load PDB " + pdbUri + ": " + err );
    },
  });
}
</script>
<div id="container-01" class="mol-container"></div>
<script>
$(function() {
  let element = $('#container-01');
  let config = { backgroundColor: 'white' };
  viewer = $3Dmol.createViewer( element, config );
});
</script>
</div>
</body>
</html>"""

Ember3D = Ember3D(args.model_checkpoint, args.t5_dir, args.device)

class StructHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        seq = self.path.split("/")[-1]
        if seq.startswith("?"):
            seq = seq[1:]
        else:
            self.send_response(200)
            self.send_header('Content-type','text/html')
            self.end_headers()
            self.wfile.write(bytes(landing_page, "utf8"))
            return

        if len(seq) == 0 or len(seq) > 2000:
            self.send_response(400)
            self.send_header('Content-type','text/plain')
            self.end_headers()
            self.wfile.write(bytes("Unsupported sequence length\n", "utf8"))
            return

        for aa in seq:
            if aa not in aa_list:
                self.send_response(400)
                self.send_header('Content-type','text/plain')
                self.end_headers()
                self.wfile.write(bytes("Invalid amino-acid: {}\n".format(aa), "utf8"))
                return

        with torch.cuda.amp.autocast():
            result = Ember3D.predict(seq)
            pdb_out = result.to_pdb(None)

        self.send_response(200)
        self.send_header('Content-type','text/plain')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(bytes(pdb_out, "utf8"))

    def do_POST(self):
        self.send_response(400)
        self.send_header('Content-type','text/plain')
        self.end_headers()
        self.wfile.write(bytes("POST requests not supported\n", "utf8"))

with HTTPServer(('', args.port), StructHandler) as server:
    print("Starting server...")
    server.serve_forever()
