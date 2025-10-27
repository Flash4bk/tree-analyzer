import 'dart:io';
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

void main() => runApp(const MyApp());

const apiBase = String.fromEnvironment('API_BASE', defaultValue: 'https://<your-railway-app>.up.railway.app');

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext c) => MaterialApp(
        title: 'Tree Analyzer',
        theme: ThemeData(colorSchemeSeed: Colors.green, useMaterial3: true),
        home: const HomePage(),
      );
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  File? _image;
  Map<String, dynamic>? _result;
  bool _loading = false;
  final _picker = ImagePicker();
  Offset? _p1;
  Offset? _p2;

  Future<void> _pickGallery() async {
    final x = await _picker.pickImage(source: ImageSource.gallery, imageQuality: 95);
    if (x != null) setState(() { _image = File(x.path); _result = null; _p1 = _p2 = null; });
  }

  Future<void> _pickCamera() async {
    final x = await _picker.pickImage(source: ImageSource.camera, imageQuality: 95);
    if (x != null) setState(() { _image = File(x.path); _result = null; _p1 = _p2 = null; });
  }

  Future<void> _send([bool withPoints = false]) async {
    if (_image == null) return;
    setState(() => _loading = true);
    try {
      final form = FormData.fromMap({
        'image': await MultipartFile.fromFile(_image!.path, filename: 'photo.jpg'),
        if (withPoints && _p1 != null && _p2 != null) ...{
          'p1x': _p1!.dx.toInt(),
          'p1y': _p1!.dy.toInt(),
          'p2x': _p2!.dx.toInt(),
          'p2y': _p2!.dy.toInt(),
        }
      });
      final dio = Dio(BaseOptions(connectTimeout: const Duration(seconds: 25), receiveTimeout: const Duration(seconds: 25)));
      final res = await dio.post('$apiBase/analyze', data: form);
      setState(() => _result = Map<String, dynamic>.from(res.data));
    } catch (e) {
      setState(() => _result = {'ok': false, 'error': e.toString()});
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final canManual = _image != null && _result?['needs_manual_scale'] == true;
    return Scaffold(
      appBar: AppBar(title: const Text('Tree Analyzer')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Row(
            children: [
              ElevatedButton.icon(onPressed: _pickCamera, icon: const Icon(Icons.photo_camera), label: const Text('Снять')),
              const SizedBox(width: 12),
              OutlinedButton.icon(onPressed: _pickGallery, icon: const Icon(Icons.photo), label: const Text('Галерея')),
              const SizedBox(width: 12),
              FilledButton.icon(onPressed: _image != null && !_loading ? () => _send(false) : null, icon: const Icon(Icons.upload), label: Text(_loading ? 'Отправка...' : 'Анализ')),
            ],
          ),
          const SizedBox(height: 16),
          if (_image != null) _ImageWithPoints(
            file: _image!,
            onPointsChanged: (a,b) { _p1=a; _p2=b; },
            enablePicking: canManual,
          ),
          const SizedBox(height: 12),
          if (canManual)
            FilledButton.icon(
              onPressed: _loading ? null : () => _send(true),
              icon: const Icon(Icons.straighten),
              label: const Text('Отправить с ручным масштабом'),
            ),
          const SizedBox(height: 16),
          if (_result != null) _ResultCard(data: _result!),
        ],
      ),
    );
  }
}

class _ImageWithPoints extends StatefulWidget {
  final File file;
  final bool enablePicking;
  final void Function(Offset?, Offset?) onPointsChanged;
  const _ImageWithPoints({required this.file, required this.onPointsChanged, required this.enablePicking});
  @override
  State<_ImageWithPoints> createState() => _ImageWithPointsState();
}

class _ImageWithPointsState extends State<_ImageWithPoints> {
  Offset? p1;
  Offset? p2;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: widget.enablePicking ? (d){
        setState(() {
          if (p1 == null) p1 = d.localPosition;
          else if (p2 == null) p2 = d.localPosition;
          else { p1 = d.localPosition; p2 = null; }
        });
        widget.onPointsChanged(p1, p2);
      } : null,
      child: Stack(
        children: [
          Image.file(widget.file),
          if (p1 != null)
            Positioned(left: p1!.dx-6, top: p1!.dy-6, child: _dot()),
          if (p2 != null)
            Positioned(left: p2!.dx-6, top: p2!.dy-6, child: _dot()),
          if (p1 != null && p2 != null)
            CustomPaint(
              painter: _LinePainter(p1!, p2!),
              child: SizedBox(width: double.infinity, height: (Image.file(widget.file).height ?? 0)),
            ),
        ],
      ),
    );
  }

  Widget _dot() => Container(width: 12, height: 12, decoration: const BoxDecoration(color: Colors.red, shape: BoxShape.circle));
}

class _LinePainter extends CustomPainter {
  final Offset a,b;
  _LinePainter(this.a,this.b);
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..strokeWidth = 3.0..style = PaintingStyle.stroke;
    canvas.drawLine(a, b, paint);
  }
  @override
  bool shouldRepaint(covariant _LinePainter old) => old.a!=a || old.b!=b;
}

class _ResultCard extends StatelessWidget {
  final Map<String, dynamic> data;
  const _ResultCard({required this.data});
  @override
  Widget build(BuildContext c) {
    if (data['ok'] != true) {
      return Card(child: Padding(
        padding: const EdgeInsets.all(16),
        child: Text('Ошибка: ${data['reason'] ?? data['error'] ?? 'unknown'}'),
      ));
    }
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: DefaultTextStyle.merge(style: const TextStyle(fontSize: 16), child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Вид: ${data['species']}', style: const TextStyle(fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            Text('Высота: ${data['height_m']} м'),
            Text('Ширина кроны: ${data['crown_width_m']} м'),
            Text('Диаметр ствола: ${data['trunk_diameter_m'] ?? '—'} м'),
            const SizedBox(height: 8),
            Text('Масштаб: ${data['scale_m_per_px'] != null ? "1 px = ${data['scale_m_per_px']} м" : "не найден"}'
            + (data['scale_source'] == 'auto' ? ' (авто)' : data['scale_source']=='manual' ? ' (ручной)' : '')),
          ],
        )),
      ),
    );
  }
}
