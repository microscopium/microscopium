function to_tsv(data_table) {
    var columns = Object.keys(data_table)
    var nrows = data_table[columns[0]].length
    var lines = [columns.join('\t')];

    for (var i = 0; i < nrows; i++) {
        var row = [];
        for (var j = 0; j < columns.length; j++) {
            var column = columns[j];
            row.push(data_table[column][i].toString());
        }
        lines.push(row.join('\t'));
    }
    filetext = lines.join('\n').concat('\n');
    return filetext
}


var filename = 'data_result.tsv';
var filetext = to_tsv(source.data)
var blob = new Blob([filetext], { type: 'text/tsv;charset=utf-8;' });
var link = document.createElement("a");
link.href = URL.createObjectURL(blob);
link.download = filename
link.target = "_blank";
link.style.visibility = 'hidden';
link.dispatchEvent(new MouseEvent('click'))
