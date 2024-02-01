import os
import json


def write_header(f):
    f.write("<html>\n")
    f.write("<head>\n")
    f.write("""<style>
table, td, th {
  border: 1px solid black;
}

table {
  width: 100%;
  border-collapse: collapse;
}
</style>""")
    f.write("</head>\n")
    f.write("<body>\n")


img_path = '/project/3dlg-hcvc/scan2arch/www/roomformer'

def generate_html():
    out_path = img_path
    
    test_json = json.load(open("/localhome/qiruiw/research/RoomFormer/data/mp3d/annotations/test.json"))
    
    with open(os.path.join(out_path, "mp3d.html"), "w") as f:
        write_header(f)
        f.write("<table>\n")
        columns = ["Scene", "Point Cloud", "Density Map", "GT floorplan", "Pred floorplan", "GT floorplan (semantic)", "Pred floorplan (semantic)"]
        f.write("<tr>%s</tr>\n" % ("".join(["<th>%s</th>" % c for c in columns])))
        f.flush()
        
        for i, image in enumerate(test_json["images"]):
            img_id = int(image["id"])
            file_name = image["file_name"]
            scene_id = file_name.split(".")[0]
            # house_id, level_id = scene_id.split('_')
            
            pc_topdown = os.path.join(f"roomformer/mp3d/{scene_id}/topdown.png")
            density_map = os.path.join(f"roomformer/data/mp3d/test/{scene_id}.png")
            gt_floorplan = os.path.join(f"roomformer/checkpoints/eval_mp3d/{img_id}_gt.png")
            pred_floorplan = os.path.join(f"roomformer/checkpoints/eval_mp3d/{img_id}_pred_floorplan.png")
            gt_sem_floorplan = os.path.join(f"roomformer/checkpoints/eval_mp3d_sem_rich/{img_id}_sem_rich_gt.png")
            pred_sem_floorplan = os.path.join(f"roomformer/checkpoints/eval_mp3d_sem_rich/{img_id}_sem_rich_pred.png")
            
            f.write("<tr>")
            f.write(f"<td style='width:10%;'>{scene_id}</td>")
            f.write(f"<td style='width:40%;'><img style='height:250px;' src='{pc_topdown}'/></td>")
            f.write(f"<td style='width:40%;'><img style='height:250px;' src='{density_map}'/></td>")
            f.write(f"<td style='width:40%;'><img style='height:250px;' src='{gt_floorplan}'/></td>")
            f.write(f"<td style='width:40%;'><img style='height:250px;' src='{pred_floorplan}'/></td>")
            f.write(f"<td style='width:40%;'><img style='height:250px;' src='{gt_sem_floorplan}'/></td>")
            f.write(f"<td style='width:40%;'><img style='height:250px;' src='{pred_sem_floorplan}'/></td>")
            f.write("</tr>")
            
        # f.write("</tr>")
        f.write("</table>\n")
        f.write("</body>\n")
        f.write("</html>\n")
        
generate_html()