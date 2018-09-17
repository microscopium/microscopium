var data = source.data;
var filetext = 'index,info,neighbors,url,x,y,path\n';
for (var i = 0; i < data['info'].length; i++) {
    var currRow = [data['index'][i].toString(),
                   data['info'][i].toString(),
                   data['neighbors'][i].toString(),
                   data['url'][i].toString(),
                   data['x'][i].toString(),
                   data['y'][i].toString(),
                   data['path'][i].toString().concat('\n')];

    var joined = currRow.join();
    filetext = filetext.concat(joined);
}

var filename = 'selected_data.csv';
var blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename);
} else {
    var link = document.createElement("a");
    link = document.createElement('a')
    link.href = URL.createObjectURL(blob);
    link.download = filename
    link.target = "_blank";
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'))
}
