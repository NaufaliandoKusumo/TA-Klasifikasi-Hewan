package com.example.bandungzoochatbot;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import com.example.bandungzoochatbot.ml.Model;


public class ClassifyActivity extends AppCompatActivity {

    Button camera, gallery;
    ImageView imageView;
    TextView result, facts_sub, description_sub;
    TextView facts1, facts2;
    TextView description;
    int imageSize = 299;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_classify);

        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);

        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);
        facts_sub = findViewById(R.id.facts_sub);
        facts1 = findViewById(R.id.facts1);
        facts2 = findViewById(R.id.facts2);
        description_sub = findViewById(R.id.description_sub);
        description = findViewById(R.id.description);

//        arrow.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//                startActivity(new Intent(getApplicationContext(), Chatbot.class));
//            }
//        });

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 100);
                }
            }
        });
    }

    public void classifyImage(Bitmap image) {
        try {
            Model model = Model.newInstance(getApplicationContext());

            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 299, 299, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;

            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();

            int maxPos = 0;
            float maxConfidence = 0;
            String[] classes = {"AYAM MUTIARA", "BABI RUSA", "BANTENG JAWA", "BEKANTAN", "BERANG-BERANG", "BERUANG MADU", "BINTURONG", "BURUNG UNTA","ELAND", "GAJAH", "HARIMAU", "JALAK BALI", "JERAPAH", "JULANG SULAWESI", "KAKATUA", "KAMBING GUNUNG", "KANGKARENG PERUT PUTIH", "KASUARI", "KIJANG", "KOMODO", "KROONKRAN", "KUDA", "KUDA NIL", "LANDAK JAWA", "LUTUNG JAWA", "MERAK", "ONTA", "ORANGUTAN", "PELIKAN", "RAKUN", "RANGKONG BADAK", "RUSA", "SIAMANG", "SINGA", "SITATUNGA", "WALABI", "WATUSI", "WILDEBEEST BIRU", "ZEBRA"};
            for (int i = 0; i < confidences.length; i++) {
                Log.i("Checkpoint", classes[i]);
                Log.i("Checkscore", String.valueOf(confidences[i]));
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
//            Log.i("maxPos", String.valueOf(maxPos));
            Log.i("classified", classes[maxPos]);
            Log.i("maxConf", String.valueOf(maxConfidence));

            AnimalsData newObj = new AnimalsData();

            String[] fact1 = newObj.getFact1();
            String[] fact2 = newObj.getFact2();
            String[] desc = newObj.getDescription();

            facts_sub.setText(R.string.facts_sub);
            description_sub.setText(R.string.description_sub);

            result.setText(classes[maxPos]);
            facts1.setText(fact1[maxPos]);
            facts2.setText(fact2[maxPos]);
            description.setText(desc[maxPos]);

            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode == RESULT_OK) {
            if (requestCode == 3) {
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            } else {
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}